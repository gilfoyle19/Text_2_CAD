import cadquery as cq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CADEnvironment:
    """Environment for CAD model generation using CadQuery operations"""
    
    # Class attributes
    STATE_SIZE = 15
    
    def __init__(self, max_operations=20, workspace_size=100):
        self.max_operations = max_operations
        self.workspace_size = workspace_size
        self.state_size = self.STATE_SIZE  # Instance attribute
        self.reset()
        
        # Define action space
        self.actions = {
            0: 'create_circle',
            1: 'create_rectangle', 
            2: 'create_polygon',
            3: 'translate',
            4: 'rotate',
            5: 'extrude',
            6: 'union',
            7: 'cut',
            8: 'finish'
        }
        
        self.action_space_size = len(self.actions)
        # Remove the duplicate state_size definition since it's now set in __init__
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_workplane = cq.Workplane("XY")
        self.solid_objects = []
        self.sketch_objects = []
        self.operation_count = 0
        self.done = False
        self.last_operation_success = True
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros(self.state_size)
        
        # Operation count normalized
        state[0] = self.operation_count / self.max_operations
        
        # Number of objects
        state[1] = len(self.solid_objects) / 10.0  # Normalize to reasonable range
        state[2] = len(self.sketch_objects) / 10.0
        
        # Last operation success
        state[3] = 1.0 if self.last_operation_success else 0.0
        
        # Bounding box info of current objects (if any)
        if self.solid_objects:
            try:
                bbox = self.solid_objects[-1].val().BoundingBox()
                state[4] = bbox.xlen / self.workspace_size
                state[5] = bbox.ylen / self.workspace_size
                state[6] = bbox.zlen / self.workspace_size
                state[7] = bbox.center.x / self.workspace_size
                state[8] = bbox.center.y / self.workspace_size
                state[9] = bbox.center.z / self.workspace_size
            except:
                pass
        
        # Available actions (simplified)
        state[10] = 1.0 if len(self.sketch_objects) > 0 else 0.0  # Can extrude
        state[11] = 1.0 if len(self.solid_objects) >= 2 else 0.0  # Can do boolean ops
        state[12] = 1.0 if len(self.solid_objects) > 0 or len(self.sketch_objects) > 0 else 0.0  # Can transform
        
        # Random features for exploration
        state[13] = np.random.random()
        state[14] = np.random.random()
        
        return state
    
    def step(self, action):
        """Execute an action and return new state, reward, done"""
        if self.done:
            return self._get_state(), 0, True, {}
        
        reward = 0
        info = {}
        
        try:
            if action == 0:  # create_circle
                reward = self._create_circle()
            elif action == 1:  # create_rectangle
                reward = self._create_rectangle()
            elif action == 2:  # create_polygon
                reward = self._create_polygon()
            elif action == 3:  # translate
                reward = self._translate()
            elif action == 4:  # rotate
                reward = self._rotate()
            elif action == 5:  # extrude
                reward = self._extrude()
            elif action == 6:  # union
                reward = self._union()
            elif action == 7:  # cut
                reward = self._cut()
            elif action == 8:  # finish
                reward = self._finish()
                self.done = True
            
            self.operation_count += 1
            
            # End episode if max operations reached
            if self.operation_count >= self.max_operations:
                self.done = True
                reward += self._calculate_final_reward()
            
        except Exception as e:
            logger.warning(f"Action {action} failed: {str(e)}")
            reward = -1
            self.last_operation_success = False
        
        return self._get_state(), reward, self.done, info
    
    def _create_circle(self):
        """Create a circle sketch"""
        radius = np.random.uniform(5, 20)
        center_x = np.random.uniform(-30, 30)
        center_y = np.random.uniform(-30, 30)
        
        sketch = cq.Workplane("XY").center(center_x, center_y).circle(radius)
        self.sketch_objects.append(sketch)
        self.last_operation_success = True
        return 2  # Reward for creating new geometry
    
    def _create_rectangle(self):
        """Create a rectangle sketch"""
        width = np.random.uniform(10, 40)
        height = np.random.uniform(10, 40)
        center_x = np.random.uniform(-30, 30)
        center_y = np.random.uniform(-30, 30)
        
        sketch = cq.Workplane("XY").center(center_x, center_y).rect(width, height)
        self.sketch_objects.append(sketch)
        self.last_operation_success = True
        return 2
    
    def _create_polygon(self):
        """Create a polygon sketch"""
        n_sides = np.random.randint(3, 8)
        radius = np.random.uniform(5, 20)
        center_x = np.random.uniform(-30, 30)
        center_y = np.random.uniform(-30, 30)
        
        sketch = cq.Workplane("XY").center(center_x, center_y).polygon(n_sides, radius)
        self.sketch_objects.append(sketch)
        self.last_operation_success = True
        return 2
    
    def _translate(self):
        """Translate existing objects"""
        if not self.solid_objects and not self.sketch_objects:
            self.last_operation_success = False
            return -1
        
        dx = np.random.uniform(-20, 20)
        dy = np.random.uniform(-20, 20)
        dz = np.random.uniform(-10, 10)
        
        # Translate the most recent object
        if self.solid_objects:
            self.solid_objects[-1] = self.solid_objects[-1].translate((dx, dy, dz))
        elif self.sketch_objects:
            self.sketch_objects[-1] = self.sketch_objects[-1].translate((dx, dy, 0))
        
        self.last_operation_success = True
        return 1
    
    def _rotate(self):
        """Rotate existing objects"""
        if not self.solid_objects and not self.sketch_objects:
            self.last_operation_success = False
            return -1
        
        angle = np.random.uniform(0, 360)
        axis_choice = np.random.choice(['X', 'Y', 'Z'])
        
        if self.solid_objects:
            if axis_choice == 'X':
                self.solid_objects[-1] = self.solid_objects[-1].rotate((0,0,0), (1,0,0), angle)
            elif axis_choice == 'Y':
                self.solid_objects[-1] = self.solid_objects[-1].rotate((0,0,0), (0,1,0), angle)
            else:
                self.solid_objects[-1] = self.solid_objects[-1].rotate((0,0,0), (0,0,1), angle)
        elif self.sketch_objects:
            # Only Z rotation for sketches
            self.sketch_objects[-1] = self.sketch_objects[-1].rotate((0,0,0), (0,0,1), angle)
        
        self.last_operation_success = True
        return 1
    
    def _extrude(self):
        """Extrude sketch to create solid"""
        if not self.sketch_objects:
            self.last_operation_success = False
            return -1
        
        height = np.random.uniform(5, 30)
        sketch = self.sketch_objects.pop()  # Remove from sketches
        solid = sketch.extrude(height)
        self.solid_objects.append(solid)
        self.last_operation_success = True
        return 3  # Higher reward for creating 3D geometry
    
    def _union(self):
        """Union two solid objects"""
        if len(self.solid_objects) < 2:
            self.last_operation_success = False
            return -1
        
        obj1 = self.solid_objects.pop()
        obj2 = self.solid_objects.pop()
        result = obj1.union(obj2)
        self.solid_objects.append(result)
        self.last_operation_success = True
        return 4  # High reward for complex operation
    
    def _cut(self):
        """Cut one solid from another"""
        if len(self.solid_objects) < 2:
            self.last_operation_success = False
            return -1
        
        obj1 = self.solid_objects.pop()
        obj2 = self.solid_objects.pop()
        result = obj2.cut(obj1)  # Cut obj1 from obj2
        self.solid_objects.append(result)
        self.last_operation_success = True
        return 4
    
    def _finish(self):
        """Finish the model"""
        if self.solid_objects:
            return 5  # Reward for completing with solid objects
        elif self.sketch_objects:
            return 2  # Lower reward for sketches only
        else:
            return -2  # Penalty for empty model
    
    def _calculate_final_reward(self):
        """Calculate final reward based on model complexity"""
        reward = 0
        
        # Reward for having solid objects
        reward += len(self.solid_objects) * 2
        
        # Small reward for sketches
        reward += len(self.sketch_objects) * 0.5
        
        # Bonus for model complexity
        if self.solid_objects:
            try:
                # Try to calculate volume as complexity measure
                total_volume = sum([obj.val().Volume() for obj in self.solid_objects])
                if total_volume > 0:
                    reward += min(total_volume / 1000, 10)  # Cap the volume bonus
            except:
                pass
        
        return reward
    
    def export_model(self, filename="generated_model.step"):
        """Export the final model"""
        if self.solid_objects:
            try:
                final_model = self.solid_objects[0]
                for obj in self.solid_objects[1:]:
                    final_model = final_model.union(obj)
                final_model.val().exportStep(filename)
                logger.info(f"Model exported to {filename}")
                return True
            except Exception as e:
                logger.error(f"Export failed: {str(e)}")
                return False
        return False


class DQN(nn.Module):
    """Deep Q-Network for CAD generation"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    """DQN Agent for learning CAD generation"""
    
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class CADTrainer:
    """Main trainer class for DRL CAD generation"""
    
    def __init__(self):
        self.env = CADEnvironment()
        # Use the state_size from the environment instance
        self.agent = DQNAgent(self.env.state_size, self.env.action_space_size)
        self.scores = []
        self.best_score = -float('inf')
        
    def train(self, episodes=1000, target_update_freq=100, save_freq=100):
        """Train the DQN agent"""
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while not self.env.done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train the agent
                if len(self.agent.memory) > 32:
                    self.agent.replay()
            
            self.scores.append(total_reward)
            
            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.agent.save(f'dqn_cad_model_{episode}.pth')
                logger.info(f"Model saved at episode {episode}")
            
            # Log progress
            if episode % 50 == 0:
                avg_score = np.mean(self.scores[-50:])
                logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                
                # Save best model
                if total_reward > self.best_score:
                    self.best_score = total_reward
                    self.agent.save('best_dqn_cad_model.pth')
    
    def test(self, episodes=10, model_path='best_dqn_cad_model.pth'):
        """Test the trained agent"""
        try:
            self.agent.load(model_path)
            self.agent.epsilon = 0  # No exploration during testing
            logger.info("Loaded trained model for testing")
        except:
            logger.warning("Could not load trained model, using current model")
        
        test_scores = []
        successful_models = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            actions_taken = []
            
            while not self.env.done:
                action = self.agent.act(state)
                actions_taken.append(self.env.actions[action])
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            
            test_scores.append(total_reward)
            
            # Try to export the model
            if self.env.export_model(f"test_model_{episode}.step"):
                successful_models += 1
            
            logger.info(f"Test Episode {episode}: Score = {total_reward:.2f}")
            logger.info(f"Actions: {actions_taken}")
        
        logger.info(f"Test Results: Average Score = {np.mean(test_scores):.2f}")
        logger.info(f"Successfully generated {successful_models}/{episodes} models")
        
        return test_scores
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.scores:
            logger.warning("No training data to plot")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(1, 2, 2)
        # Moving average
        window = 50
        if len(self.scores) >= window:
            moving_avg = [np.mean(self.scores[i:i+window]) for i in range(len(self.scores)-window+1)]
            plt.plot(moving_avg)
            plt.title(f'Moving Average (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    
    def generate_model(self, model_path='best_dqn_cad_model.pth', filename='generated_cad_model.step'):
        """Generate a single CAD model using the trained agent"""
        try:
            self.agent.load(model_path)
            self.agent.epsilon = 0
            logger.info("Loaded trained model for generation")
        except:
            logger.warning("Could not load trained model, using current model")
        
        state = self.env.reset()
        actions_taken = []
        
        while not self.env.done:
            action = self.agent.act(state)
            action_name = self.env.actions[action]
            actions_taken.append(action_name)
            state, reward, done, _ = self.env.step(action)
            logger.info(f"Action: {action_name}, Reward: {reward}")
        
        # Export the final model
        success = self.env.export_model(filename)
        
        logger.info(f"Model generation completed")
        logger.info(f"Actions taken: {actions_taken}")
        logger.info(f"Final model {'exported successfully' if success else 'export failed'}")
        
        return success, actions_taken


def main():
    """Main function to run the DRL CAD generator"""
    # Create trainer
    trainer = CADTrainer()
    
    # Check if we should train or test
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'train'
    
    if mode == 'train':
        logger.info("Starting training mode")
        trainer.train(episodes=500)
        trainer.plot_training_progress()
        
    elif mode == 'test':
        logger.info("Starting test mode")
        trainer.test(episodes=5)
        
    elif mode == 'generate':
        logger.info("Starting generation mode")
        trainer.generate_model()
        
    else:
        logger.info("Available modes: train, test, generate")
        logger.info("Usage: python drl_cad_generator.py [mode]")


if __name__ == "__main__":
    main()

# Save this script as: drl_cad_generator.py