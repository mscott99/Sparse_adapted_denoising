import torch
from typing import Optional
import torch.nn as nn
import torch.optim as optim
from torch.optim import lbfgs
from sparse_recov.convvae.utils import get_model


class NeuralNetInverter:
    def __init__(self, forward_net: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inverter for a given feedforward neural network.

        Args:
            forward_net (nn.Module): The feedforward neural network to invert.
            device (str): Device to run the computation on.
        """
        assert hasattr(forward_net, "layers")
        assert isinstance(forward_net.layers, list), "layers must be a list of nn.Module"
        assert hasattr(forward_net, "input_shape")
        assert isinstance(forward_net.input_shape, tuple), "input_shape must be a tuple"
        self.forward_net = forward_net.to(device).eval()
        self.device = device
        # Assuming layers is a list of layer functions
        self.layers = forward_net.layers
        self.input_shape=forward_net.input_shape
        self.intermediate_shapes = self._compute_intermediate_shapes(
            torch.zeros(*forward_net.input_shape))
        self.offsets=self._initialize_offsets()

    def _compute_intermediate_shapes(self, dummy_input: torch.Tensor) -> list[tuple]:
        """
        Compute the shapes of activations between layers using a dummy input.

        Args:
            dummy_input (torch.Tensor): Dummy input tensor to pass through the network.

        Returns:
            List[Tuple]: List of shapes of intermediate activations.
        """
        shapes=[]
        x=dummy_input.to(self.device)
        for layer in self.layers:
            assert isinstance(layer, nn.Module)
            x=layer(x)
            shapes.append(x.shape)
        return shapes[:-1]  # Exclude the final output shape

    def _initialize_offsets(self) -> nn.ParameterList:
        """
        Initialize offset parameters for each intermediate activation space.

        Returns:
            nn.ParameterList: List of offset tensors as learnable parameters.
        """
        offsets=nn.ParameterList()
        for shape in self.intermediate_shapes:
            offset=nn.Parameter(torch.zeros(
                shape, device=self.device), requires_grad=True)
            offsets.append(offset)
        return offsets

    def forward_with_offsets(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with added offsets.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the network after applying offsets.
        """
        for i, (layer, offset) in enumerate(zip(self.layers, self.offsets)):
            x=layer(x)
            if i < len(self.offsets):  # Apply offset after each layer except the last
                x=x + offset
        return x

    def compute_loss(self, input_tensor: torch.Tensor, target_output: torch.Tensor, l2_lambda: float) -> torch.Tensor:
        """
        Compute the total loss including MSE and L2 regularization on offsets.

        Args:
            input_tensor (torch.Tensor): Input tensor to the network.
            target_output (torch.Tensor): Target output to match.
            l2_lambda (float): Current L2 regularization strength for offsets.

        Returns:
            torch.Tensor: Total loss value.
        """
        output=self.forward_with_offsets(input_tensor)
        mse_loss=torch.sum((output-target_output)**2)
        l2_loss=0.0
        for offset in self.offsets:
            l2_loss += torch.sum(offset ** 2)
        print("mse_loss: ", mse_loss.item())
        print("offset_loss: ", l2_loss)
        total_loss=mse_loss + l2_lambda * l2_loss
        return total_loss

    def invert(self, target_output: torch.Tensor, initial_input: Optional[torch.Tensor]=None,
               initial_l2_lambda: float=1e-6, max_l2_lambda: float=1e3, num_increases: int=10,
               lbfgs_steps_per_stage: int=3) -> torch.Tensor:
        """
        Invert the neural network to recover an input matching the target output.

        Args:
            target_output (torch.Tensor): Target output to match.
            initial_input (torch.Tensor, optional): Initial guess for input. Defaults to random.
            initial_l2_lambda (float): Starting L2 regularization strength.
            max_l2_lambda (float): Maximum L2 regularization strength.
            num_increases (int): Number of times to increase L2 strength.
            lbfgs_steps_per_stage (int): Number of LBFGS steps per L2 strength stage.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Recovered input and final offsets.
        """
        # Initialize input tensor as a learnable parameter
        if initial_input is None:
            input_tensor=nn.Parameter(torch.randn(
                self.input_shape, device=self.device), requires_grad=True)
        else:
            input_tensor=nn.Parameter(
                initial_input.clone().to(self.device), requires_grad=True)

        # Compute the factor for exponential increase of L2 regularization strength
        l2_increase_factor=(max_l2_lambda / initial_l2_lambda) ** (
            1.0 / (num_increases - 1)) if num_increases > 1 else 1.0
        current_l2_lambda=initial_l2_lambda

        # Parameters to optimize: input tensor and offsets
        params=[input_tensor] + list(self.offsets)

        def total_offset_norm():
            return sum(torch.sum(offset ** 2) for offset in self.offsets)

        for stage in range(num_increases):
            print(
                f"Stage {stage + 1}/{num_increases}, L2 Lambda: {current_l2_lambda:.6f}")
            # Set up LBFGS optimizer for the current stage
            optimizer=optim.LBFGS(
                params, max_iter=lbfgs_steps_per_stage, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                loss=self.compute_loss(
                    input_tensor, target_output, current_l2_lambda)
                loss.backward()
                return loss

            # Run LBFGS optimization for the specified number of steps
            for _ in range(lbfgs_steps_per_stage):
                optimizer.step(closure)

            # Exponentially increase the L2 regularization strength
            current_l2_lambda *= l2_increase_factor

        # Final optimization without offsets (drop offsets completely)
        print("Final optimization without offsets")
        params=[input_tensor]  # Optimize only the input
        optimizer=optim.LBFGS(
            params, max_iter=lbfgs_steps_per_stage, line_search_fn="strong_wolfe")

        def final_closure():
            optimizer.zero_grad()
            # Forward pass without offsets
            output=self.forward_net(input_tensor)
            loss=torch.sum((output-target_output)**2)
            loss.backward()
            return loss

        # Run final optimization steps without offsets
        for _ in range(lbfgs_steps_per_stage):
            optimizer.step(final_closure)

        return input_tensor.detach()


class VAEGeneratorWrapper(nn.Module):
    def __init__(self, model_label: str,
                 device: str='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Wrapper class to extract the decoder (generator) part of a pretrained VAE model
        and make it compatible with NeuralNetInverter.

        Args:
            pretrained_path (str): Path to the pretrained VAE model weights.
            z_dim (int): Dimension of the latent space (z_dim of the VAE).
            input_shape (Tuple[int, int, int]): Shape of the input image (C, H, W) for the original VAE.
            device (str): Device to run the computation on.
        """
        super().__init__()
        self.device=device
        # Original input shape of the VAE (for encoder)
        # Input shape for the decoder (latent space dimension)

        # Extract the decoder components
        self.layers=self._load_decoder_layers(model_label)
        self.to(device)

    def _load_decoder_layers(self, model_label: str) -> list[nn.Module]:
        # Initialize a temporary VAE model to load the weights into
        temp_vae=get_model(model_label)
        temp_vae.to(self.device).eval()
        self.vae=temp_vae
        self.input_shape = (1, temp_vae.z_dim,)

        layers=[]
        layers.extend(list(self.vae.decoder_linear))
        class ReshapeLayer(nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape=target_shape
        
            def forward(self, x):
                if len(x.shape)==1:
                    return x.view(*temp_vae.conv_out_shape[1:])
                else:
                    return x.view(x.shape[0], *temp_vae.conv_out_shape[1:])
        reshape_layer=ReshapeLayer(self.vae.conv_out_shape)
        layers.append(reshape_layer)
        layers.extend(list(self.vae.decoder_conv))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x=layer(x)
        return x

if __name__ == "__main__":
    # Test with model label vae-32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_label = "vae-32"
    
    # Initialize the VAE generator wrapper (decoder)
    vae_generator = VAEGeneratorWrapper(model_label, device)
    print("VAE Generator Wrapper initialized.")

    # Initialize the inverter for the VAE decoder
    inverter = NeuralNetInverter(vae_generator, device)
    print("Neural Net Inverter initialized.")

    # Create a dummy target output (e.g., a random image)
    dummy_code = torch.randn(1,32).to(device)  # Adjust shape to match model output
    dummy_target = vae_generator(dummy_code)
    print("Dummy target output created.")

    # Run the inversion process
    recovered_input = inverter.invert(target_output=dummy_target, initial_l2_lambda=0.01, max_l2_lambda=1e5, lbfgs_steps_per_stage=5)
    print("Inversion completed.")
    print("Recovered input shape:", recovered_input.shape)
    print("Inversion precision: ", torch.norm(vae_generator(recovered_input)- dummy_target)/torch.norm(dummy_target))
