loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(5):
    for batch, (X, y) in enumerate(train_d):
        model.train()
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
