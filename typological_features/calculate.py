import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np


def run(target_language, feature_index, train_epoch, args=None):
    torch.manual_seed(1)

    x_train = torch.ones((1, args.input_size)).to(args.device)
    y_train = torch.ones(1).to(args.device)

    feature = args.features[feature_index]
    for l in args.languages:
        if l not in [target_language] and feature in list(args.INFO[l]):
            x0 = args.all_embeddings[l]
            value = args.INFO[l][feature]
            y0 = torch.ones(1000) * args.features2num[feature][value] #change value to number of lines!!
            x_train = torch.cat((x_train, x0.to(args.device)), 0)
            y_train = torch.cat((y_train, y0.to(args.device)), )

    x_train = x_train[1:].type(torch.FloatTensor)
    y_train = y_train[1:].type(torch.LongTensor)

    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    x_test = args.all_embeddings[target_language].to(args.device)
    y_test = np.ones(1000) * args.features2num[feature][value] #change value to number of lines!!

    output_size = len(args.features2num[feature].keys())
    net = torch.nn.Sequential(
        torch.nn.Linear(args.input_size, args.hidden_dim),
        torch.nn.Dropout(args.hidden_dropout_prob),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, output_size),
    )
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # 优化器
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(train_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            optimizer.zero_grad()
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            loss.backward()
            optimizer.step()

    net.eval()
    out = net(x_test)
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.detach().cpu().numpy().squeeze()

    # print("accuracy:", np.sum(pred_y==y_test)/10000)
    return np.sum(pred_y == y_test) / 1000 #change value to number of lines!!
