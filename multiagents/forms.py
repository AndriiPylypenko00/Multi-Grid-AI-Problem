from django import forms


class InputForm(forms.Form):
    num_of_neurons = forms.IntegerField(label="Number of neurons", initial=20)
    num_of_cookies = forms.IntegerField(label="Number of food", initial=10)
    num_of_layers = forms.IntegerField(label="Number of layers", initial=2)
    num_of_agents = forms.IntegerField(label="Number of agents", initial=10)
    size_x = forms.IntegerField(label="X size of board", initial=20)
    size_y = forms.IntegerField(label="Y size of board", initial=20)
    mutation_prob = forms.FloatField(label="Propability of mutation", initial=0.1)
    cookie_prize = forms.IntegerField(label="Food prize", initial=5)
    border_punishment = forms.IntegerField(label="Border punishment", initial=20)
    move_punishment = forms.IntegerField(label="Move punishment", initial=1)
    num_of_epochs = forms.IntegerField(label="Number of epochs", initial=500)
    num_moves = forms.IntegerField(label="Number of moves per epoch", initial=25)
    epochs_per_save = forms.IntegerField(label="Save every N epochs", initial=100)
    number_of_episodes = forms.IntegerField(label="Number of episodes for each epoch", initial=25)
