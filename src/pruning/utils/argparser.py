import argparse


mod_choices = ["VGG16", "ResNet34", "ResNet50"]
dataset_choices = ["CIFAR10", "CIFAR100"]
layer_wise = [True, False]

def parsing():

    parser = argparse.ArgumentParser(description="Arguments to run a specific model on a specific dataset")
    parser.add_argument('--model', choices=mod_choices, default = "VGG16", help='chosen model'+'Options:'+str(mod_choices))
    parser.add_argument('--dataset', choices=dataset_choices, default = "CIFAR10", help='chosen dataset'+'Options:'+str(dataset_choices))
    parser.add_argument('--prunetype', choices = layer_wise, default = False, help = 'pruning type_global or layer_wise'+ str(layer_wise))
    parser.add_argument('--epochs',  default = 70, help = 'number of epochs')
    parser.add_argument('--lr', default = 0.1, help= 'learning rate')
    parser.add_argument('--retain', default= 0.05, help= 'how much weights are retained')
    parser.add_argument('--step', default= 20, help = 'step_size')
    parser.add_argument('--weightdecay', default = 0.0005, help='weight_decay')
    parser.add_argument('--batchsize', default = 128, help = 'batch_size')

    return parser
