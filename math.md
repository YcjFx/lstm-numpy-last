$d\text{IFOG}[t, :, 3n{out}:] = (1 - \text{IFOGf}[t, :, 3n{out}:]^2) \cdot d\text{IFOGf}[t, :, 3n{out}:]$

$y = \text{IFOGf}[t, :, :3n{out}]$

$d\text{IFOG}[t, :, :3n_{out}] = [y \cdot (1.0 - y)] \cdot d\text{IFOGf}[t, :, :3n{out}]$