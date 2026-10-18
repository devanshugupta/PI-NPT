"""Physics-informed training of MLP and attention-between-datapoints (NPT) surrogates for
convection, diffusion and reaction PDEs, scored against closed-form solutions."""
from .evaluate import errors, fields, predict
from .models import MLP, NPT, Solution, make
from .pde import COEFF, Problem, grid, sample_collocation
from .train import TrainConfig, context_size, curriculum_path, fit, fit_curriculum

__all__ = ["Problem", "COEFF", "grid", "sample_collocation", "MLP", "NPT", "Solution", "make", "TrainConfig",
           "context_size", "curriculum_path", "fit", "fit_curriculum", "errors", "fields", "predict"]
