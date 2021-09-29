import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path", default="./tusimple")
    parser.add_argument("--save", required=False, help="Directory to save model checkpoint", default="./checkpoints")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=10)
    parser.add_argument("--bs", required=False, type=int, help="Batch size", default=2)
    parser.add_argument("--val", required=False, type=bool, help="Use validation", default=False)
    parser.add_argument("--lr", required=False, type=float, help="Learning rate", default=0.00005)
    parser.add_argument("--pretrained", required=False, default=None, help="pretrained model path")
    parser.add_argument("--out_path", default="./output", help="output image folder")
    parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")
    return parser.parse_args()
#usercode
def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=5)
    parser.add_argument("--out_path", default="./output", help="output image folder")
    parser.add_argument("--bs", required=False, type=int, help="Batch size", default=2)
    return parser.parse_args()