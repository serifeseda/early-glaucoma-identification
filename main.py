"""
Main file to train CNN and NN models on two data sets
"""
# Copyright (C) 2018  Serife Seda Kucur

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from train_module import train_cnn, train_nn

# Train and evaluate the CNN model on BD data set
dataset  = 'BD'
dirname  = 'results/CNN/{}'.format(dataset) # this can be changed as you wish
train_cnn(dataset, dirname)

# Train and evaluate the CNN model on RT data set
dataset  = 'RD'
dirname  = 'results/CNN/{}'.format(dataset) # This can be changed as you wish
train_cnn(dataset, dirname)

# Train and evaluate the NN model on BD data set
dataset  = 'BD'
dirname  = 'results/NN/{}'.format(dataset) # this can be changed as you wish
train_nn(dataset, dirname)

# Train and evaluate the NN model on RT data set
dataset  = 'RD'
dirname  = 'results/NN/{}'.format(dataset) # This can be changed as you wish
train_nn(dataset, dirname)



