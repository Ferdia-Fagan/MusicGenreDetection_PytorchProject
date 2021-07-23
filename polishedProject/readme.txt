

First of all, install requirements.txt

Then:

1)
run from this folder:
python -m src.Main

2)
Wait till its done

3) 
In each folder [CNN_1D, CNN_2D, LSTM] go into each and run the jupyter notebook 
Demonstrate.ipynb

You can view the graphs for each, and see best training, validation, and final test , accuracy.

4) 
After running each of the Demonstrate.ipynb notebook, you can go into the respective "tensorboardsLogs" folder. This will be created in for example "/CNN_1D/tensorboardLogs/".
within this folder enter command:

tensorboard --logdir .

You will then be prompted with a url after some output from the command.
Open url in browser and you will be able to view the models I made (all parameters etc).
I had to do this as I lost my project with all my diagrams and comments(as already explained).
So... I hope tensorboard will do because I dont have enough time to draw them again :'(













