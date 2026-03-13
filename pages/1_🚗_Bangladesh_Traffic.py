import streamlit as st
import sys
import os
import torch
import torch.nn as nn

# Add the autonomous_vehicle directory to the path
project_dir = os.path.join(os.path.dirname(__file__), "autonomous_vehicle")
sys.path.insert(0, project_dir)

# Define CBAM classes BEFORE importing anything else
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Register CBAM with ultralytics BEFORE loading the model
try:
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules.block as block

    tasks.CBAM = CBAM
    block.CBAM = CBAM  # Register in the block module where it's expected
    block.ChannelAttention = ChannelAttention
    block.SpatialAttention = SpatialAttention
except Exception as exc:
    st.error("Failed to import project dependencies for Bangladesh Traffic page.")
    st.code(str(exc), language="text")
    st.info("If deployed on Streamlit Cloud, pin Python to 3.11 using runtime.txt and redeploy.")
    st.stop()

# Now execute the main project file
main_file = os.path.join(project_dir, "main.py")
with open(main_file, encoding='utf-8') as f:
    code = f.read()
    exec(code, globals())
