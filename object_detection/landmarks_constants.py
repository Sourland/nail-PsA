WRIST = 0

# Thumb landmarks
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

# Index finger landmarks
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8

# Middle finger landmarks
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12

# Ring finger landmarks
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16

# Pinky finger landmarks
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

landmarks_per_finger = {
    'THUMB': [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    'INDEX': [INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP,
              INDEX_FINGER_TIP],
    'MIDDLE': [MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP,
               MIDDLE_FINGER_TIP],
    'RING': [RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP],
    'PINKY': [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]
}

# Create a base dictionary for joint neighbours
joint_neighbours_left_hand = {
    INDEX_FINGER_MCP: [MIDDLE_FINGER_MCP, THUMB_CMC],
    INDEX_FINGER_PIP: [MIDDLE_FINGER_PIP, THUMB_MCP],
    INDEX_FINGER_DIP: [MIDDLE_FINGER_DIP, THUMB_IP],
    INDEX_FINGER_TIP: [MIDDLE_FINGER_TIP, THUMB_TIP],

    MIDDLE_FINGER_MCP: [RING_FINGER_MCP, INDEX_FINGER_MCP],
    MIDDLE_FINGER_PIP: [RING_FINGER_PIP, INDEX_FINGER_PIP],
    MIDDLE_FINGER_DIP: [RING_FINGER_DIP, INDEX_FINGER_DIP],
    MIDDLE_FINGER_TIP: [RING_FINGER_TIP, INDEX_FINGER_TIP],

    RING_FINGER_MCP: [PINKY_MCP, MIDDLE_FINGER_MCP],
    RING_FINGER_PIP: [PINKY_PIP, MIDDLE_FINGER_PIP],
    RING_FINGER_DIP: [PINKY_DIP, MIDDLE_FINGER_DIP],
    RING_FINGER_TIP: [PINKY_TIP, MIDDLE_FINGER_TIP],

    PINKY_MCP: [RING_FINGER_MCP],
    PINKY_PIP: [RING_FINGER_PIP],
    PINKY_DIP: [RING_FINGER_DIP],
    PINKY_TIP: [RING_FINGER_TIP],
}


def flip_joints_for_right(joints_dict):
    flipped = {}

    # Iterate over every joint and its neighbors in the left-hand dictionary
    for joint, neighbors in joints_dict.items():
        # For the new neighbors, swap the order of the neighbors.
        # If the left-hand joint had neighbors [A, B], the right-hand joint will have [B, A]
        new_neighbors = neighbors[::-1] if isinstance(neighbors, list) else neighbors

        flipped[joint] = new_neighbors

    return flipped


joint_neighbours_right_hand = flip_joints_for_right(joint_neighbours_left_hand)
