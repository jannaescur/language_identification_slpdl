# Hyperparameters optimization approach

This was our first approach and it has ended up being the one that achieved better results. <br>
We started from the baseline provided, and analyzed the behaviour of the network and how could be improved.<br><br>

After training the baseline we realized that the network was learning well. We did not need more capacity for the amount
of data provided. The problem was that it was overfitting. In fact, this is a problem that we are not being able to completely 
solve in any of the approaches. In order to try to reduce the overfitting we implemented some solutions:<br>

- Use more training data (in our case, this was not possible as the dataset was given in the Kaggle competition)
- Make it harder for the network to learn (in order to generalize better):
  - Change network architecture:<br><br>
  Instead of one Linar layer, we used two of them with 1024 neurons:<br>
        *self.h2o = torch.nn.Sequential(torch.nn.Linear(hidden_size, 1024),
                                      torch.nn.Dropout(),
                                      torch.nn.Linear(1024, output_size))*
  
  
  - Use Dropout:<br>
  We used Dropout inbetween the two previous layers, with the default percentage of neurons connections deactivations, which is of 50%.
  - Add regularization / weight decay:<br>
  Without reducing the network size or increase the amount of training data, adding regularization we can reduce overfitting.
   Regularization basically adds the penalty as model complexity increases. Regularization parameter (lambda) penalizes all 
   the parameters except intercept so that model generalizes the data and won’t overfit.<br>
   
    *all_linear1_params = torch.cat([x.view(-1) for x in model.h2o.parameters()])
            l2_regularization = 0.01 * torch.norm(all_linear1_params, 2)
            loss+=l2_regularization*<br>
            
    where in this case lambda is 0.01.
    
  - Reduce the size of the hidden layer:<br>
  In order to make it harder to memorize, we can also reduce the size of the hidden layer, although in this approach we used
  the same as the baseline (hidden size = 256)
    
  - Progressively reduce the learning rate:<br>
  Finally, we also add a decay in the learning rate (using LRScheduler):<br>
  *scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma = 0.5)*<br>
  
    So with these parameters the learning rate is reduced by 0.5 every 8 epochs.


We tried to change the hyperparameters thinking about the specific problem and not randomly. We got an accuracy of 93,8% 
with the 20% of the testing set.<br>
Even we could keep trying different hyperparameters, we thought that it was more interesting to try and learn completely
different approaches, which can be found in the other directories of the repository.
