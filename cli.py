from command import parser
from command.infor import infor
from command.unit import unit
from command.train import train

args = parser.parse_args()
infor(args)
unit(args)
train(args)