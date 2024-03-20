from command import parser, build_folder
from command.infor import infor
from command.unit import unit
from command.train import train

build_folder()

args = parser.parse_args()
infor(args)
unit(args)
train(args)