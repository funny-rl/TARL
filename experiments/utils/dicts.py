random_reward_dict = {
    "CliffWalking" : -108.9,  
    "ZigZag" : -111.5,
    "Bridge" : -111.9,
    "Pendulum-v1" : -1225.555,
    "LunarLander-v3": -211.427,
    "PointMaze_Large_Diverse_GDense-v3" : 9.652
}

max_score_dict = {
    "CliffWalking" : -13.0,
    "Bridge" : -13.0,
    "ZigZag" : -20.0,
    "Pendulum-v1" : -158.125,
    "LunarLander-v3": 158.709,
    "PointMaze_Large_Diverse_GDense-v3" : 63.852
}

D_MODEL_DICT = {
    "CliffWalking" : {
        "EQL" : [
            {"RARe(ours)": "EQL_epsilon"}, 
            {"RARe-M(ours)": "EQL_skip"}
        ],
        "TempoRL" : [
            {"TempoRL": "TempoRL"}, 
            {"TempoRL-M": "TempoRL_skip"}
        ],
        "UTE" : [
            {"UTE": "UTE"}, 
            {"UTE-M": "UTE_skip"}
        ],
        "null" : [
            {"DDQN": "DDQN"}
        ]
    },
    "Bridge" : {
        "EQL" : [
            {"RARe(ours)": "EQL_reverse"}, 
            {"RARe-M(ours)": "EQL_skip"}
        ],
        "TempoRL" : [
            {"TempoRL": "TempoRL"}, 
            {"TempoRL-M": "TempoRL_skip"}
        ],
        "UTE" : [
            {"UTE": "UTE"}, 
            {"UTE-M": "UTE_skip"}
        ],
        "null" : [
            {"DDQN": "DDQN"}
        ]
    },
    "ZigZag" : {
        "EQL" : [
            {"RARe(ours)": "EQL_reverse"}, 
            {"RARe-M(ours)": "EQL_skip"}
        ],
        "TempoRL" : [
            {"TempoRL": "TempoRL"}, 
            {"TempoRL-M": "TempoRL_skip"}
        ],
        "UTE" : [
            {"UTE": "UTE"}, 
            {"UTE-M": "UTE_skip"}
        ],
        "null" : [
            {"DDQN": "DDQN"}
        ]
    }
} 



ENV_DICT = {
    "CliffWalking" : {
        "pits_list" : [
            [5,1],
            [5,2],
            [5,3],
            [5,4],
            [5,5],
            [5,6],
            [5,7],
            [5,8],
            
            [4,1],
            [4,2],
            [4,3],
            [4,4],
            [4,5],
            [4,6],
            [4,7],
            [4,8],
        ],
        "shape" : [6, 10],
        "start" : [5, 0],
        "goal" : (5, 9),
    },

    "Bridge" : {
        "pits_list" : [
            [0,2],
            [1,2],
            [0,3],
            [1,3],
            [0,4],
            [1,4],
            [0,5],
            [1,5],
            [0,6],
            [1,6],
            [0,7],
            
            [1,7], # upper block
            [4,2],
            [5,2],
            [4,3],
            [5,3],
            [4,4],
            [5,4],
            [4,5],
            [5,5],
            [4,6],
            [5,6],
            [4,7],
            [5,7],

        ],
        "shape" : [6, 10],
        "start" : [0, 0],
        "goal" : (0, 9),
    },
    "ZigZag" : {
        "pits_list" : [
            [0,2],
            [1,2],
            [2,2],
            [3,2],
            [0,3],
            [1,3],
            [2,3],
            [3,3],

            [2,6],
            [3,6],
            [4,6],
            [5,6],
            [2,7],
            [3,7],
            [4,7],
            [5,7],
        ],
        "shape" : [6, 10],
        "start" : [0, 0],
        "goal" : (5, 9),
    }
}