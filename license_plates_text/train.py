import os
import copy
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import torch
from sklearn import metrics
from rich.console import Console
from rich import print

import engine
from utils.logging_config import setup_logging, general_table, predictions_table
from model.model import CRNN
from utils.plot import plot_acc, plot_losses
from utils.model_decoders import decode_predictions, decode_padded_predictions
from utils.data_loading import build_dataloaders


console = Console()

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run_training(cfg):
    """
    Function to run the training process for the text recognition model.

    Parameters:
        cfg (OmegaConf): Configuration object containing the hyperparameters and settings for the training process.

    Returns:
        None

    Description: This function sets up the logging, builds the dataloaders, initializes the model, optimizer, and scheduler,
                 trains the model, logs the progress, and saves the best model checkpoint.
    """
    log = setup_logging()

    log.info(f"[C]Configurations:\n{OmegaConf.to_yaml(cfg)}")
    print(f"[C]Configurations:\n{OmegaConf.to_yaml(cfg)}")

    train_loader, test_loader, test_original_targets, classes = build_dataloaders(cfg)

    print(f"[T]Dataset number of classes: {len(classes)}")
    print(f"[T]Classes are: {classes}")
    log.info(f"Dataset number of classes: {len(classes)}")
    log.info(f"Classes are: {classes}")

    use_cuda = torch.cuda.is_available() and cfg.basic.use_cuda
    device = torch.device(0 if use_cuda else "cpu")
    model = CRNN(
        resolution=(cfg.text_recognition.processing.image_width, cfg.text_recognition.processing.image_height),
        dims=cfg.text_recognition.model.dims,
        num_chars=len(classes),
        use_attention=cfg.text_recognition.model.use_attention,
        use_ctc=cfg.model.text_recognition.use_ctc,
        grayscale=cfg.model.text_recognition.gray_scale,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.8, patience=5, verbose=True)

    training_classes = ["∅"]
    training_classes.extend(classes)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    start = datetime.now()

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []

    for epoch in range(cfg.text_recognition.training.num_epochs):
        train_loss = engine.train_fn(model, train_loader, optim, device)
        train_loss_data.append(train_loss)
        valid_preds, test_loss = engine.eval_fn(model, test_loader, device)
        valid_loss_data.append(test_loss)
        valid_captcha_preds = []

        for valid_pred in valid_preds:
            if model.use_ctc:
                current_preds = decode_predictions(valid_pred, training_classes)
            else:
                current_preds = decode_padded_predictions(valid_pred, training_classes)
            valid_captcha_preds.extend(current_preds)

        combined = list(zip(test_original_targets, valid_captcha_preds))
        if cfg.text_recognition.bools.VIEW_INFERENCE_WHILE_TRAINING:
            table = predictions_table()
            for idx in combined:
                if cfg.text_recognition.bools.DISPLAY_ONLY_WRONG_PREDICTIONS:
                    if idx[0] != idx[1]:
                        table.add_row(idx[0], idx[1])
                else:
                    table.add_row(idx[0], idx[1])
            console.print(table)

        accuracy = metrics.accuracy_score(test_original_targets, valid_captcha_preds)
        accuracy_data.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            log.info(f"New best accuracy achieved at epoch {epoch}. Best accuracy now is: {best_acc}")
            best_model_wts = copy.deepcopy(model.state_dict())
            if cfg.text_recognition.bools.SAVE_CHECKPOINTS:
                torch.save(model, f"logs/checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        table = general_table()
        table.add_row(str(epoch), str(train_loss), str(test_loss), str(accuracy), str(best_acc))
        console.print(table)
        log.info(f"Epoch {epoch}:    Train loss: {train_loss}    Test loss: {test_loss}    Accuracy: {accuracy}")

    log.info(f"Finished training. Best Accuracy was: {(best_acc*100):.2f}%")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.text_recognition.paths.save_model_as)
    log.info(f"Saving model on {cfg.text_recognition.paths.save_model_as}\nTraining time: {datetime.now()-start}")
    plot_losses(train_loss_data, valid_loss_data)
    plot_acc(accuracy_data)


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        run_training()
    except Exception:
        console.print_exception()
