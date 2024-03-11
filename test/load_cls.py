import sys
sys.path.append("../../")

import brain
topo = brain.get_cls("Topo")(edges = None)
print(topo)