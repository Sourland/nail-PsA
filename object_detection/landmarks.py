from collections import namedtuple

Landmarks = namedtuple('Landmarks', [
    'WRIST',
    'THUMB_CMC',
    'THUMB_MCP',
    'THUMB_IP',
    'THUMB_TIP',
    'INDEX_FINGER_MCP',
    'INDEX_FINGER_PIP',
    'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP',
    'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP',
    'RING_FINGER_PIP',
    'RING_FINGER_DIP',
    'RING_FINGER_TIP',
    'PINKY_MCP',
    'PINKY_PIP',
    'PINKY_DIP',
    'PINKY_TIP'
])

# Access the named tuple values
print(Landmarks.THUMB_TIP)  # Landmarks.THUMB_TIP
# print(Landmarks.THUMB_TIP.value)  # 4

# Iterate over the named tuple values
# for landmark in Landmarks:
#     print(landmark)  # Landmarks.WRIST, Landmarks.THUMB_CMC, Landmarks.THUMB_MCP, ...

# Use the named tuple values
landmarks = Landmarks(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

