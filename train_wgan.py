from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from models.dcgan import Generator, Discriminator as Critic, weights_init
from torch.optim import RMSprop, Adam
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.nn.utils.clip_grad import clip_grad_norm_    #right?
from datasets import Datasets
from torch import Tensor
from typing import Dict, Sequence, Union
from metrics.fid_infinity import FIDInfinity
from metrics.is_infinity import ISInfinity
from torchmetrics import FID, IS, KID
from utility.utils import adjust_dimensions, normalize_to_0_1, normalize_to_0_255, parse_none_true_false, save_images_to_disk
from torchmetrics.utilities.data import dim_zero_cat
import yaml
import math

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class TrainWGAN(PyTorchTrial):
    """
    Entry point for training a Wasserstein Generative Adversarial Network (with or without Gradient Penalty).
    Initializes generator and discriminator network, loss functions, metrics and optimizers.
    Parses configuration from associated config file and global config file.
    Args:
        context:
            A PyTorchTrialContext object
    """
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.data_config = self.context.get_data_config()
        self.img_dim = tuple(self.data_config['dim'])
        self.dataset_name = self.data_config['dataset']
        self.global_parameters = self.data_config['global_parameters']

        # Load Hyperparameters from config file
        self.batch_size = self.context.get_per_slot_batch_size()
        self.learning_rate = self.context.get_hparam('lr')
        self.beta1 = self.context.get_hparam('b1')
        self.beta2 = self.context.get_hparam('b2')
        self.latent_dim = self.context.get_hparam('latent_dim')
        self.lambda_term = self.context.get_hparam('gradient_penaty_lambda')
        self.use_gradient_penalty = self.context.get_hparam('gradient_penalty')
        self.critic_iterations = self.context.get_hparam('critic_iterations')
        self.evaluate_while_trainig = self.context.get_hparam('evaluate_while_trainig')

        with open(self.data_config['global_parameters'], 'r') as file:
            try:
                data = yaml.safe_load(file)
                if self.evaluate_while_trainig:
                    self.training_metrics = data['training_metrics']
                self.evaluation_metrics = data['evaluation_metrics']
                self.save_images_config = data['save_images']
            except yaml.YAMLError:
                print(f"Couldn't load metrics from {self.data_config['global_parameters']}")

        # Map strings to class instances
        if self.evaluate_while_trainig:
            self.training_metrics = list(map(lambda data: self.context.to_device(globals()[data[0]](**parse_none_true_false(data[1]))), self.training_metrics.items()))
        
        self.save_real_images = self.save_images_config['real_images']['save']
        self.save_fake_images = self.save_images_config['fake_images']['save']

        if self.save_real_images:
            self.number_of_real_images_to_save = self.save_images_config['real_images']['number_of_images']
            self.real_images_to_save_path = self.save_images_config['real_images']['path']
            self.real_images_to_save_name = self.save_images_config['real_images']['file_name']

        if self.save_fake_images:
            self.number_of_fake_images_to_save = self.save_images_config['fake_images']['number_of_images']
            self.fake_images_to_save_path = self.save_images_config['fake_images']['path']
            self.fake_images_to_save_name = self.save_images_config['fake_images']['file_name']
            

        self.dataset_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.datasets = Datasets(self.dataset_directory, self.batch_size)
        self.tb_writer = TorchWriter().writer

        # Initialize Networks
        self.generator = self.context.wrap_model(Generator(self.latent_dim, self.img_dim))
        self.critic = self.context.wrap_model(Critic(self.img_dim, self.batch_size, use_sigmoid=False))
        self.generator.apply(weights_init)
        self.critic.apply(weights_init)

        # Initialize Optimizers
        if self.use_gradient_penalty:
            self.optimizer_gen = self.context.wrap_optimizer(Adam(self.generator.parameters(), lr=self.learning_rate, betas=[self.beta1, self.beta2]))
            self.optimizer_critic = self.context.wrap_optimizer(Adam(self.critic.parameters(), lr=self.learning_rate, betas=[self.beta1, self.beta2]))
        else:
            self.optimizer_gen = self.context.wrap_optimizer(RMSprop(self.generator.parameters(), lr=self.learning_rate))
            self.optimizer_critic = self.context.wrap_optimizer(RMSprop(self.critic.parameters(), lr=self.learning_rate))


        # Initialize Loss Functions
        if self.use_gradient_penalty:
            self.loss_func_critic = lambda real_imgs, fake_imgs: -(torch.mean(self.critic(real_imgs)) - torch.mean(self.critic(fake_imgs))) + self.lambda_term * self.gradient_penalty(self.critic, real_imgs, fake_imgs)
        else:
            self.loss_func_critic = lambda real_imgs, fake_imgs: -(torch.mean(self.critic(real_imgs)) - torch.mean(self.critic(fake_imgs)))

        self.loss_func_gen = lambda prob: -torch.mean(prob)


        if self.evaluate_while_trainig:
            self.training_scores = dict()
            for m in self.training_metrics:
                self.training_scores[type(m).__name__] = float("NaN")



    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        '''Calculate training losses and metrics on a single batch. Periodically writes data to TensorBoard.
        Args:
            batch:
                A single batch containing training data and labels
            epoch_idx:
                Index of training epoch
            batch_idx:
                Index of batch
        Returns:
            Dictionary containing calculated losses and metrics for this particular batch
        '''
        real_imgs, _ = batch

        self.generator.train()
        self.critic.train()

        self.generator.requires_grad_(True)
        self.critic.requires_grad_(False)

        # Create random noise 
        random_noise = self.context.to_device(torch.randn((self.batch_size, self.latent_dim, 1, 1)))
        # Generate Images
        fake_imgs = self.generator(random_noise)

        # Train Generator only every self.critic_iterations iterations
        if batch_idx % self.critic_iterations == 0:
            self.generator_loss = self.loss_func_gen(self.critic(fake_imgs))
            self.context.backward(self.generator_loss)
            self.context.step_optimizer(self.optimizer_gen)

        #Train Critic
        self.generator.requires_grad_(False)
        self.critic.requires_grad_(True)
        critic_loss = self.loss_func_critic(real_imgs, fake_imgs.detach()) 
        self.context.backward(critic_loss)

        # Weight Clipping
        if self.use_gradient_penalty:
            self.context.step_optimizer(self.optimizer_critic)
        else:
            self.context.step_optimizer(self.optimizer_critic, clip_grads = lambda params: clip_grad_norm_(params, 0.01))


        # Write to Tensorboard
        if batch_idx % 100 == 0:
            self.tb_writer.add_scalar("generator loss", self.generator_loss)
            self.tb_writer.add_scalar("critic loss", critic_loss)
            tb_imgs = fake_imgs.view(self.batch_size, *self.img_dim)[0:16]
            tb_imgs = adjust_dimensions(tb_imgs)
            tb_imgs = normalize_to_0_255(tb_imgs).to(dtype=torch.uint8)
            img_grid = make_grid(tb_imgs)

            self.tb_writer.add_image(f'raw: epoch index: {epoch_idx}, batch index: {batch_idx}', img_grid)

            
            if self.evaluate_while_trainig:
                with torch.no_grad():
                    real_imgs, fake_imgs = adjust_dimensions(real_imgs), adjust_dimensions(fake_imgs)
                    real_imgs_0_255, fake_imgs_0_255 = normalize_to_0_255(real_imgs).to(dtype=torch.uint8), normalize_to_0_255(fake_imgs).to(dtype=torch.uint8)
                    real_imgs_0_1, fake_imgs_0_1 = normalize_to_0_1(real_imgs), normalize_to_0_1(fake_imgs)


                    for m in self.training_metrics:
                        if isinstance(m, (FID, KID)):
                            # 0-255, uint8
                            m.update(real_imgs_0_255, real=True)
                            m.update(fake_imgs_0_255, real=False)
                        elif isinstance(m, FIDInfinity):
                            # 0-1, float32
                            m.update(real_imgs_0_1, real=True)
                            m.update(fake_imgs_0_1, real=False)
                        elif isinstance(m, ISInfinity):
                            # 0-1, float32
                            m.update(fake_imgs_0_1)
                        else:
                            # 0-255, uint8
                            m.update(fake_imgs_0_255)

                    for m in self.training_metrics:
                        if isinstance(m, KID):
                            if len(dim_zero_cat(m.real_features)) >= 5000:
                                score, _ = m.compute()
                            else:
                                score = float("NaN")
                        elif isinstance(m, (ISInfinity, FIDInfinity)):
                            if m.get_number_of_features() >= 5000:
                                score = m.compute()
                            else: 
                                score = float("NaN")
                        elif isinstance(m, IS):
                            score, _ = m.compute()
                        else:
                            score = m.compute()

                        self.training_scores[type(m).__name__] = score

            
        losses = {
                "generator_loss": self.generator_loss,
                "discriminator_loss": critic_loss,
            }

        if self.evaluate_while_trainig:
            return {**self.training_scores, **losses}
        else:
            return losses

    
    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        '''Calculate metrics on full evaluation dataset. Save images to disk if activated in settings.
        Args:
            data_loader:
                DataLoader containing evaluation dataset
        Returns:
            Dictionary containing calculated metrics for the evaluation dataset
        '''
        real_data_to_disk = []

        evaluation_metrics = list(map(lambda data: self.context.to_device(globals()[data[0]](**parse_none_true_false(data[1]))), self.evaluation_metrics.items()))
        scores = dict()

        for imgs, _ in data_loader:
            real_imgs = self.context.to_device(imgs)

            # Create random noise 
            random_noise = self.context.to_device(torch.randn(real_imgs.shape[0], self.latent_dim, 1, 1))
            # Generate Images
            fake_imgs = self.generator(random_noise)

            real_imgs, fake_imgs = adjust_dimensions(real_imgs), adjust_dimensions(fake_imgs)
            real_imgs_0_255, fake_imgs_0_255 = normalize_to_0_255(real_imgs).to(dtype=torch.uint8), normalize_to_0_255(fake_imgs).to(dtype=torch.uint8)
            real_imgs_0_1, fake_imgs_0_1 = normalize_to_0_1(real_imgs), normalize_to_0_1(fake_imgs)

            if self.save_real_images:
                if len(real_data_to_disk) == 0 or len(dim_zero_cat(real_data_to_disk)) < self.number_of_real_images_to_save:
                    real_data_to_disk.append(real_imgs_0_255)


            for m in evaluation_metrics:
                if isinstance(m, (FID, KID)):
                    # 0-255, uint8
                    m.update(real_imgs_0_255, real=True)
                    m.update(fake_imgs_0_255, real=False)
                elif isinstance(m, FIDInfinity):
                    # 0-1, default
                    m.update(real_imgs_0_1, real=True)
                    m.update(fake_imgs_0_1, real=False)
                elif isinstance(m, ISInfinity):
                    # 0-1, default
                    m.update(fake_imgs_0_1)
                else:
                    # 0-255, uint8
                    m.update(fake_imgs_0_255)


        for m in evaluation_metrics:
            if isinstance(m, KID):
                if len(dim_zero_cat(m.real_features)) >= 5000:
                    score, _ = m.compute()
                else:
                    score = float("NaN")
            elif isinstance(m, (ISInfinity, FIDInfinity)):
                if m.get_number_of_features() >= 5000:
                    score = m.compute()
                else: 
                    score = float("NaN")
            elif isinstance(m, (IS, KID)):
                score, _ = m.compute()
            else:
                score = m.compute()

            if not math.isnan(score):
                scores[type(m).__name__] = score


        if self.save_fake_images:
            disk_imgs = self.generator(self.context.to_device(torch.randn(self.number_of_fake_images_to_save, self.latent_dim, 1, 1)))
            disk_imgs = adjust_dimensions(disk_imgs)
            disk_imgs = normalize_to_0_255(disk_imgs).to(dtype=torch.uint8)
            save_images_to_disk(disk_imgs, path=self.fake_images_to_save_path, file_name=self.fake_images_to_save_name)

        if self.save_real_images:
            real_data_to_disk = dim_zero_cat(real_data_to_disk)[:self.number_of_real_images_to_save]
            save_images_to_disk(real_data_to_disk, path=self.real_images_to_save_path, file_name=self.real_images_to_save_name)
        
        return scores


    def gradient_penalty(self, critic: nn.Module, real_imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        epsilon = self.context.to_device(
            torch.randn(real_imgs.shape[0], 1, 1, 1)
            .expand_as(real_imgs))
        new_imgs = (epsilon * real_imgs + (1 - epsilon) * fake_imgs).requires_grad_()
        new_imgs_prob = critic(new_imgs)
        
        grads = torch.autograd.grad(
            outputs=new_imgs_prob,
            inputs=new_imgs, 
            grad_outputs=torch.ones_like(new_imgs_prob), 
            create_graph=True, 
            retain_graph=True)[0]
        grads = grads.view(grads.shape[0], -1)
        return torch.mean(torch.pow(grads.norm(2, dim=1) - 1, 2))


    def build_training_data_loader(self) -> DataLoader:
        """Build DataLoader for training dataset
        Returns:
            DataLoader
        """
        return self.datasets.get_data_loader(self.dataset_name, train=True)


    def build_validation_data_loader(self) -> DataLoader:
        """Build DataLoader for evaluation dataset
        Returns:
            DataLoader
        """
        dataloader = self.datasets.get_data_loader(self.dataset_name, train=False)
        self.number_of_evaluation_batches = len(dataloader)
        return dataloader