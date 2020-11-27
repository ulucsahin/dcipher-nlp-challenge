import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import test

class Trainer(object):
    # keep track of values
    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.

    def __init__(self, config, train_dataset, test_dataset):
        self.loss_fn = F.cross_entropy
        self.config = config
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def train_model(self,model, optim, epoch):
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        # train on gpu
        if torch.cuda.is_available():
            model.cuda()

        steps = 0
        model.train()
        last_idx = math.floor(len(self.train_dataset) / self.config.batch_size)
        train_acc_epoch = 0
        for idx, batch in enumerate(dataloader):
            # if dataset exhausted, reset dataloader and reshuffle (safeguard)
            if(idx == last_idx - 1):
                dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

            title = batch[0] # shape: (batch_size, title_pad_length, token_amount, embedding_size)
            abstract = batch[1] # shape: (batch_size, abstract_pad_length, token_amount, embedding_size)
            target = batch[2] # shape: (batch_size)
            target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                title = title.cuda()
                abstract = abstract.cuda()
                target = target.cuda()

            # convert to float tensor
            title = title.float()
            abstract = abstract.float()

            optim.zero_grad()
            prediction = model(title, abstract)
            loss = self.loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects / len(batch[0])
            loss.backward()
            self.clip_gradient(model, 1e-1)
            optim.step()
            steps += 1

            train_acc_epoch += acc.item()

            if steps % self.config.print_frequency == 0:
                print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        train_acc_epoch /= last_idx - 1

        # evaluate
        test_acc_epoch = test.evaluate_model(model, self.test_dataset, self.config.batch_size)
        print(f"Accuracy at the end of epoch {epoch}: {test_acc_epoch}")

        # save values at the end of each epoch for plotting
        self.train_accs.append(train_acc_epoch)
        self.test_accs.append(test_acc_epoch)
        # self.plot()

        if epoch % 1 == 0:
            # Save model and values
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'parameters': (epoch),
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'test_accs': self.test_accs,
                'best_test_acc': self.best_test_acc
            }, f'model/trained_{epoch}.pth')
            print(f'Epoch {epoch}, model successfully saved.')

        # save separately if best model
        if (test_acc_epoch > self.best_test_acc):
            self.best_test_acc = test_acc_epoch
            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'parameters': (epoch),
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'test_accs': self.test_accs,
                'best_test_acc': self.best_test_acc
            }, f'model/BestModel.pth')
            print(f'Epoch {epoch}, *BestModel* successfully saved.')



    def plot(self):
        # plt.plot(range(len(self.train_losses)), self.train_losses, label="line 1")
        plt.plot(range(len(self.train_accs)), self.train_accs, label="Train Accuracy")
        plt.plot(range(len(self.test_accs)), self.test_accs, label="Test Accuracy")
        plt.xlabel('Epoch')
        # Set the y axis label of the current axis.
        plt.ylabel('Accuracy')
        # Set a title of the current axes.
        plt.title('Accuracy over epochs on train and test datasets.')
        # show a legend on the plot
        plt.legend()
        # Display a figure.
        plt.show()

        # plt.plot('x', 'y1', data=self.train_accs, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
        # plt.plot('x', 'y2', data=self.train_losses, marker='', color='olive', linewidth=2)
        # plt.legend()
        # plt.show()



