import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import nn
from d2l import torch as d2l

from utils.HPO import Trainer, SoftmaxRegression


class HPOTrainer(Trainer):
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_id = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_id += 1
        return 1 - accuracy / val_batch_id

def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config['learning_rate']
    trainer = HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=64)
    model = SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return trainer.validation_error().detach().numpy()



if __name__ == "__main__":
    config_space = {"learning_rate": stats.loguniform(1e-4, 1)}

    errors, values = [], []
    num_iterations = 5

    for i in range(num_iterations):
        learning_rate = config_space["learning_rate"].rvs()
        print(f"Trial {i}: learning_rate = {learning_rate}")
        y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
        print(f"    validation_error = {y}")
        values.append(learning_rate)
        errors.append(y)

    best_idx = np.argmin(errors)
    print(f"optimal learning rate = {values[best_idx]}")
    plt.show()


















