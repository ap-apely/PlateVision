import torch

def decode_padded_predictions(predictions, classes, pad_token="∅"):
    """
    Decode padded predictions into a list of strings by replacing the indices with their corresponding class names.

    Args:
        predictions (torch.Tensor): A tensor of predictions with shape (batch_size, sequence_length, num_classes).
        classes (list): A list of classes, where each class is a string.
        pad_token (str, optional): The padding token used in the predictions. Defaults to "∅".

    Returns:
        texts (list): A list of strings, where each string is the decoded prediction for a sample in the batch.
    """
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()

    output_texts = []
    for prediction in predictions:
        string = ""
        for idx in prediction:
            if idx == 0:  
                break  
            string += classes[idx]
        output_texts.append(string)

    return output_texts

def decode_predictions(predictions, classes, blank_token="∅"):
    """
    CTC RNN Layer decoder.

    1. Collapse repeating digits into a single instance unless separated by a blank token.
    2. Collapse multiple consecutive blanks into one blank.

    Args:
        predictions (torch.Tensor): A tensor of predictions with shape (batch_size, sequence_length, num_classes).
        classes (list): A list of classes, where each class is a string.
        blank_token (str, optional): The token used by CTC to represent empty space. Defaults to "∅".

    Returns:
        texts (list): A list of strings, where each string is the decoded prediction for a sample in the batch.
    """
    predictions = predictions.permute(1, 0, 2)
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()
    texts = []
    for i in range(predictions.shape[0]):
        string = ""
        batch_e = predictions[i]

        for j in range(len(batch_e)):
            string += classes[batch_e[j]]

        string = string.split(blank_token)
        string = [x for x in string if x != ""]
        string = [list(set(x))[0] for x in string]
        texts.append("".join(string))
    return texts
