import torch
import torch.nn.functional as F

def test(model, testset_loader, device):
    """Evaluate a model on a test set.

    Args:
        model (nn.Module): The model to evaluate.
        testset_loader (DataLoader): A DataLoader instance containing the test set.
        device (torch.device): The device (GPU or CPU) to run the evaluation on.

    Returns:
        None
    """
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))