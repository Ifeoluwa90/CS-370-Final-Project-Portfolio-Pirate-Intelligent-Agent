# CS 370 Final Project Portfolio: Pirate Intelligent Agent
## Treasure Hunt Game with Deep Q-Learning

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

---

## 📋 Project Overview

This repository contains the final project for **CS 370: Current and Emerging Trends in Computer Science**, focusing on artificial intelligence and machine learning. The project implements an intelligent pirate agent that learns to navigate a treasure hunt maze using **Deep Q-Learning** and **Neural Networks**.

### 🎯 Project Goals
- Develop an AI agent capable of optimal pathfinding in a maze environment
- Implement deep reinforcement learning algorithms from scratch
- Achieve 100% win rate across all possible starting positions
- Demonstrate understanding of exploration vs. exploitation in AI systems

---

## 🏗️ System Architecture

### Environment
- **Maze Size**: 8×8 grid world
- **Agent**: Pirate character seeking treasure
- **Actions**: 4 directional movements (up, down, left, right)
- **Objective**: Navigate from any starting position to treasure location

### AI Implementation
- **Algorithm**: Deep Q-Learning with Experience Replay
- **Neural Network**: 3-layer feedforward architecture
- **Training Strategy**: Epsilon-greedy exploration with adaptive decay
- **Performance Metric**: Win rate optimization reaching 100% success

---

## 🚀 Key Features

### ✨ Technical Implementation
- **Deep Q-Learning**: Reinforcement learning with neural network function approximation
- **Experience Replay**: Efficient learning through stored episode replay
- **Adaptive Exploration**: Dynamic epsilon-greedy strategy optimization
- **State Representation**: Flattened grid encoding for neural network input

### 📊 Performance Achievements
- **100% Win Rate**: Achieved optimal performance across all starting positions
- **Efficient Training**: Converged to optimal policy in ~1000 epochs
- **Robust Navigation**: Successfully handles complex maze configurations
- **Real-time Visualization**: Interactive maze state display during training

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.7+ | Core programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.x | Neural network API |
| **NumPy** | Latest | Numerical computations |
| **Matplotlib** | Latest | Visualization and plotting |
| **Jupyter** | Latest | Interactive development environment |

---

## 📖 Installation & Setup

### Prerequisites
```bash
Python 3.7 or higher
Jupyter Notebook
Virtual environment (recommended)
```

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CS370-Final-Project.git
   cd CS370-Final-Project
   ```

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv cs370_env
   source cs370_env/bin/activate  # On Windows: cs370_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install tensorflow keras numpy matplotlib jupyter
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook TreasureHuntGame.ipynb
   ```

---

## 🎮 How to Run

### Training the Agent
1. Open `TreasureHuntGame.ipynb` in Jupyter
2. Execute cells sequentially to:
   - Load environment and dependencies
   - Initialize neural network model
   - Train the agent using deep Q-learning
   - Visualize training progress
   - Test final performance

### Key Code Sections
```python
# Build neural network model
model = build_model(maze)

# Train using deep Q-learning
qtrain(model, maze, epochs=1000, max_memory=512, data_size=32)

# Test performance
completion_check(model, qmaze)
play_game(model, qmaze, pirate_start)
```

---

## 📈 Results & Performance

### Training Metrics
- **Final Win Rate**: 100% (32/32 successful completions)
- **Training Duration**: ~25 minutes (1000 epochs)
- **Convergence**: Achieved 90%+ win rate by epoch 376
- **Memory Efficiency**: 512 experience replay buffer

### Performance Visualization
The agent demonstrates optimal pathfinding by:
- Finding shortest routes to treasure
- Avoiding obstacles and boundaries
- Adapting to different starting positions
- Maintaining consistent high performance

---

## 🧠 Learning Outcomes

This project demonstrates mastery of:

### Core AI Concepts
- **Reinforcement Learning**: Understanding of reward-based learning systems
- **Neural Networks**: Deep learning architecture design and implementation
- **Optimization**: Gradient-based learning and loss minimization

### Advanced Techniques
- **Experience Replay**: Memory-efficient learning from past episodes
- **Exploration-Exploitation**: Balanced strategy for optimal learning
- **State Representation**: Effective encoding of environmental information

### Software Engineering
- **Modular Design**: Clean separation of environment, agent, and training logic
- **Documentation**: Comprehensive code comments and technical analysis
- **Testing**: Validation of agent performance across diverse scenarios

---

## 🔬 Technical Analysis

### Algorithm Deep Dive
The implementation uses **Deep Q-Learning** with the following key components:

1. **Neural Network Architecture**:
   - Input Layer: 64 neurons (8×8 maze flattened)
   - Hidden Layers: 2 layers × 64 neurons with PReLU activation
   - Output Layer: 4 neurons (action Q-values)

2. **Training Strategy**:
   - Epsilon-greedy exploration (ε = 0.1, reduced to 0.05)
   - Adam optimizer with MSE loss function
   - Batch training with 32 experience samples

3. **Experience Replay**:
   - Circular buffer storing 512 episodes
   - Random sampling for training stability
   - Breaks temporal correlation in training data

---

## 🚧 Future Enhancements

### Potential Improvements
- [ ] **Larger Mazes**: Scale to more complex environments
- [ ] **Dynamic Obstacles**: Handle changing maze configurations
- [ ] **Multi-Agent**: Implement competitive treasure hunting
- [ ] **Transfer Learning**: Apply learned policies to new maze layouts
- [ ] **Visualization**: Enhanced real-time training visualizations

### Advanced Features
- [ ] **Dueling DQN**: Implement dueling network architecture
- [ ] **Prioritized Replay**: Weight important experiences more heavily
- [ ] **Double DQN**: Reduce overestimation bias in Q-learning

---

## 📝 Academic Context

### Course Information
- **Course**: CS 370 - Current and Emerging Trends in Computer Science
- **Institution**: Southern New Hampshire University
- **Focus**: Artificial Intelligence and Machine Learning

### Project Requirements Met
- ✅ Deep Q-Learning implementation from scratch
- ✅ Neural network architecture design
- ✅ Pathfinding problem solution
- ✅ 100% performance achievement
- ✅ Comprehensive technical analysis

---

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📄 License

This project is created for academic purposes as part of CS 370 coursework. Please respect academic integrity guidelines when referencing this work.

---

## 🙏 Acknowledgments

- **TensorFlow Team**: For the deep learning framework
- **OpenAI**: For reinforcement learning research inspiration
- **Academic Community**: For foundational research in AI and machine learning

---

## 📞 Contact

**Student**: Ifeoluwa Adewoyin  
**Email**: Ifeoluwaadewoyin90@gmail.com 
**Course**: CS 370  
**Project**: Pirate Intelligent Agent  

---

*"Artificial Intelligence is not just about creating smart machines; it's about understanding intelligence itself."*

**⭐ Star this repository if you found it helpful for your AI learning journey!**
