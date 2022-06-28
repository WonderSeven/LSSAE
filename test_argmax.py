import torch
if __name__ == '__main__':
    pred = torch.rand([10, 2]).float()
    targets = torch.tensor([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    pred_y = torch.argmax(pred, dim=-1)

    acc = torch.sum(pred_y == targets) / pred.size(0)
    print(acc)