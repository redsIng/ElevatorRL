classdef Elevator < rl.env.MATLABEnvironment
    % RLENVDOUBLEINTEGRATORABSTRACT: Creates abstract class for Elevator
    % control
    
    properties
        
        % time step
        Ts = 1
        %Gravity
        G = 9.8
        % Final Point, will terminate if absolute position is exceeded
        FinalPoint = 5
        
        % Acceleration
        A = 9.8;
        % Goal threshold, will terminate if norm of state is less than
        % threshold
        GoalThreshold = 5e-2
        %Velocity along y-axis
        V0y = 0
        % Lower Bound y-axis
        lby = 0
        % Upper Bound y-axis
        uby = 8
        % Lower Bound vy
        lbv = -2
        % Upper Bound vy
        ubv = 8
        %Reward
        R = -1
        % K:= elevator acceleration
        K = 0.3
        %Enabling Plotting Env Observer
        PlotValue
    end
    properties (Access = protected)
        MaxForce_ = Inf
    end
    properties (Dependent)
        % Max force
        MaxForce
    end
    properties
        % system state [s,ds]'
        State = zeros(1,2)
    end
    properties (Transient,Access = public)
        Visualizer = []
    end
    properties (Access = private)
        
        reward double = -1
        acceleration double = 0
        
    end
    methods (Abstract, Access = protected)
        force = getForce(this,force);
    end
    
    methods (Access = protected)
        function setMaxForce_(this,val)
            % define how the setting of max force will behave, which is
            % different among continuous/discrete implementations of the
            % environment
            this.MaxForce_ = val;
            this.ActionInfo.Values = [-val,val];
        end
        
    end
    methods
        function this = Elevator(ActionInfo)
            % Define observation info
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'states';
            ObservationInfo.Description = 's, ds';
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            %             this.lby = -2;
            %             this.uby = this.FinalPoint + fix(this.FinalPoint/2)+1;
        end
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',2},'','State');
            this.State = state(:);
            if this.PlotValue == 1
                notifyEnvUpdated(this);
            end
        end
        function set.GoalThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','GoalThreshold');
            this.GoalThreshold = val;
        end
        function set.FinalPoint(this,d)
            validateattributes(d,{'numeric'},{'finite','real','positive','scalar'},'','FinalPoint');
            this.FinalPoint = d;
            notifyEnvUpdated(this);
        end
        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = ElevatorVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
        end
        function set.MaxForce(this,val)
            validateattributes(val,{'numeric'},{'real','positive','scalar'},'','MaxForce');
            setMaxForce_(this,val);
        end
        function val = get.MaxForce(this)
            val = this.MaxForce_;
        end
        
        function set.V0y(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','V0y');
            this.V0y = val;
        end
        function set.PlotValue(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','PlotValue');
            this.PlotValue = val;
        end
        function set.A(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','V0y');
            this.A = val;
        end
        function set.R(this,val)
            validateattributes(val,{'numeric'},{'real','<=',0,'size',[1 1]},'','R');
            this.R = val;
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        
        function resetEnv(this)
            %             this.State = [0,0];
            this.V0y = 0;
        end
        
        function [sp,rwd,isTerminal] = step(this,s,action,plotValue)
            this.PlotValue = plotValue;
            %Action
            %             action = getForce(this,action);
            % Time Step
            t = this.Ts;
            % Updating Enviroment State
            this.State = s;
            %Get next state
            % We have to describe the elevator dynamics along x-axis and
            % y-axis.
            % x-axis: for each time step x(t) = x0
            % y-axis: for each time step y(t) = y0+v0y*t+1/2*ay*(t)^2
            yt = this.State(1);
            v0t = this.State(2);
            at = this.A+action;
            %             atp = at - this.G;
            atp = action;
            ytp = yt+v0t*t+0.5*atp*(t)^2;
            if ytp < this.lby || ytp >= this.uby
                this.resetEnv()
                this.A = 9.8;
                rwd = -1;
                sp = [0;0];
                this.State = sp;
                isTerminal = false;
                return
            end
            
            ytp = min(max(ytp,this.lby),this.uby);
            this.V0y = min(max(v0t+atp*t,this.lbv),this.ubv);
            
            sp = [ytp;this.V0y];
            if norm(sp) < this.GoalThreshold
%                 this.resetEnv()
%                 this.A = 9.8;
                rwd = -1;
                sp = [0;0];
                this.State = sp;
                isTerminal = false;
                return
            end
            this.State = sp;
            
            this.A = at;
            %             The episode will terminate under the following conditions:
            %             1. the mass moves more than X units away from the origin
            %             2. the norm of the state is less than some threshold
            %
            %             The second point is critical for training as it prevents the
            %             replay buffer being saturated with 0s for training
            isdone = abs(sp(2)) == this.FinalPoint ;%|| norm(sp) < this.GoalThreshold ;
            rwd = this.R;
            if isdone == 1
                isTerminal = true;
            else
                isTerminal = false;
            end
        end
        
    end
end
