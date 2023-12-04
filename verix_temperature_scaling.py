import torch
from torch import nn, optim
from torch.nn import functional as F
from .ece import ECELoss

class ModelWithVeriXTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithVeriXTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))# * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits, explanation_sizes):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature and explanation sizes to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        explanation_sizes = explanation_sizes.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature / torch.add(explanation_sizes, 1)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, explanation_sizes, max_iter):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        explanation_sizes = explanation_sizes.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits, explanation_sizes), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits, explanation_sizes), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits, explanation_sizes), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
    
    def test(self, test_loader, explanation_sizes):
        print("Testing")
        self.cuda()
        explanation_sizes = explanation_sizes.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the test set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in test_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits, explanation_sizes), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits, explanation_sizes), labels).item()
        print('Temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))



