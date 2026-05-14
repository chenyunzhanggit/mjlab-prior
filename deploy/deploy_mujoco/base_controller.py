from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Base controller class for all policy controllers
    
    This abstract base class defines the common interface and functionality
    that all controller implementations should follow.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base controller
        """
        self.num_actions = None
        # Required attributes that must be set by subclasses
        self.num_obs = None
        self.kps = None
        self.kds = None
        self.action_scale = None
        self.default_dof_pos = None
        
        # Validate that subclasses properly initialize required attributes
        self._validate_required_attributes()
        
    def _validate_required_attributes(self):
        """Validate that all required attributes are properly set by subclasses"""
        required_attrs = ['num_obs', 'kps', 'kds', 'action_scale', 'default_dof_pos']
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise NotImplementedError(f"Subclass must set '{attr}' attribute")
        
    @abstractmethod
    def reset(self):
        """
        Reset controller state
        This method should reset all controller internal states to initial values
        """
        pass
    
    @abstractmethod
    def step(self, mujoco_data):
        """
        Compute control action for one step
        
        Args:
            mujoco_data: MuJoCo data object containing current simulation state
            
        Returns:
            tuple: (target_q, kps, kds)
                - target_q: Target joint positions (29,) array
                - kps: Joint stiffness parameters (29,) array  
                - kds: Joint damping parameters (29,) array
        """
        pass
    
    def __str__(self):
        # TODO
        """String representation of controller"""
        pass    
    
    def __repr__(self):
        # TODO
        """Detailed representation of controller"""
        pass