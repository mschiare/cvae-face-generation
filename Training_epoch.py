from LossFunction import kl_loss_function, loss_function
import torch

def training_epoch(model, optimizer, dataloader, beta):
    device = 'cuda'
    model.train()
    total_loss=0.0
    for x, target in dataloader:

        optimizer.zero_grad()

        # target ha dimensioni [batch_size, 40]
        male = target[:, 20].unsqueeze(1) # [batch_size] -> [batch_size,1]
        smiling= target[:, 31].unsqueeze(1) # [batch_size] -> [batch_size,1]
        young = target[:, 39].unsqueeze(1) # [batch_size] -> [batch_size,1]

        cond = torch.cat([male, smiling, young], dim=1).to(dtype=torch.float, device=device)  # [batch_size,3] 
        


        x=x.to(device=device)

        output, mu, log_sigma=model(x, cond)

        loss=loss_function(output, x, mu, log_sigma, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print('Epoch completed.  loss=', avg_loss)
