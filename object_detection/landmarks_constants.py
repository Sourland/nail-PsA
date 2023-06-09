import numpy as np

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8

MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12

RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

joint_neighbours_left_hand = {

    INDEX_FINGER_MCP: MIDDLE_FINGER_MCP,
    INDEX_FINGER_PIP: MIDDLE_FINGER_PIP,
    INDEX_FINGER_DIP: MIDDLE_FINGER_DIP,
    INDEX_FINGER_TIP: MIDDLE_FINGER_TIP,

    MIDDLE_FINGER_MCP: [INDEX_FINGER_MCP, RING_FINGER_MCP],
    MIDDLE_FINGER_TIP: [INDEX_FINGER_TIP, RING_FINGER_TIP],
    MIDDLE_FINGER_DIP: [INDEX_FINGER_DIP, RING_FINGER_DIP],
    MIDDLE_FINGER_PIP: [INDEX_FINGER_PIP, RING_FINGER_PIP],

    RING_FINGER_MCP: [MIDDLE_FINGER_MCP, PINKY_MCP],
    RING_FINGER_TIP: [MIDDLE_FINGER_TIP, PINKY_TIP],
    RING_FINGER_DIP: [MIDDLE_FINGER_DIP, PINKY_DIP],
    RING_FINGER_PIP: [MIDDLE_FINGER_PIP, PINKY_PIP],

    PINKY_MCP: RING_FINGER_MCP,
    PINKY_TIP: RING_FINGER_TIP,
    PINKY_DIP: RING_FINGER_DIP,
    PINKY_PIP: RING_FINGER_PIP,
}

joint_neighbours_right_hand = {

    INDEX_FINGER_MCP: MIDDLE_FINGER_MCP,
    INDEX_FINGER_PIP: MIDDLE_FINGER_PIP,
    INDEX_FINGER_DIP: MIDDLE_FINGER_DIP,
    INDEX_FINGER_TIP: MIDDLE_FINGER_TIP,

    MIDDLE_FINGER_MCP: [RING_FINGER_MCP, INDEX_FINGER_MCP],
    MIDDLE_FINGER_TIP: [RING_FINGER_TIP, INDEX_FINGER_TIP],
    MIDDLE_FINGER_DIP: [RING_FINGER_DIP, INDEX_FINGER_DIP],
    MIDDLE_FINGER_PIP: [RING_FINGER_PIP, INDEX_FINGER_PIP],

    RING_FINGER_MCP: [PINKY_MCP, MIDDLE_FINGER_MCP],
    RING_FINGER_TIP: [PINKY_TIP, MIDDLE_FINGER_TIP],
    RING_FINGER_DIP: [PINKY_DIP, MIDDLE_FINGER_DIP],
    RING_FINGER_PIP: [PINKY_PIP, MIDDLE_FINGER_PIP],

    PINKY_MCP: RING_FINGER_MCP,
    PINKY_TIP: RING_FINGER_TIP,
    PINKY_DIP: RING_FINGER_DIP,
    PINKY_PIP: RING_FINGER_PIP,
}

areas_of_interest = [
    INDEX_FINGER_TIP, INDEX_FINGER_DIP, INDEX_FINGER_PIP,
    MIDDLE_FINGER_TIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_PIP,
    RING_FINGER_TIP, RING_FINGER_DIP, RING_FINGER_PIP,
    PINKY_TIP, PINKY_DIP, PINKY_PIP
]
