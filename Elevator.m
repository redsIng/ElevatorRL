classdef Elevator < rl.env.MATLABEnvironment
    % RLENVDOUBLEINTEGRATORABSTRACT: Creates abstract class for Elevator
    % control
    
    properties
        % GAIN
        Gain = 1.0
        % time step
        Ts = 0.1
        %Gravity
        G = 9.8
        % Final Point, will terminate if absolute position is exceeded
        FinalPoint = 5
        % Force to be applied
        act = 0;
        % Acceleration
        A = 9.8;
        % Goal threshold, will terminate if norm of state is less than
        % threshold
        GoalThreshold = 5e-2
        %Velocity along y-axis
        V0y = 0
        % Lower Bound y-axis
        lby = -2
        % Upper Bound y-axis
        uby = 8
        % Lower Bound vy
        lbv = -2
        % Upper Bound vy
        ubv = 8
        %Enabling Plotting Env Observer
        PlotValue=0
        % Reward Weights (in continuous time to remove dependence on sample
        % time)
        % reward =  - integral(x'*Q*x + u'*R*u)
        Q  = diag([10 1]) %infinite horizon Q, used to find Qd
        R  = 0.01
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
        % reward = - sum_i(x_i'*Qd*x_i + u_i'*Rd*u_i)
        % Note, these weights are derived from Q and R and will be updated
        % after Q and R are set
        Qd (2,2) double = eye(2)
        Rd (1,1) double = 0.01
        Nd (2,1) double = [0 0]'
        
        % discrete system matrices
        Ad (2,2) double
        Bd (2,1) double
        Cd (2,2) double
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
        
        function updatePerformanceWeights(this)
            % get the continuous linearized system
            a = [0 1;0 0];
            b = [0;this.Gain];
            
            % Determine discrete equivalent of continuous cost function
            % along with Ad,Bd matrices of discretized system. This is
            % equivalent to the discretization in lqrd
            q = this.Q;
            r = this.R;
            nn = [0 0]';
            Nx = 2; Nu = 1;
            n = Nx+Nu;
            Za = zeros(Nx); Zb = zeros(Nx,Nu); Zu = zeros(Nu);
            M = [ -a' Zb   q  nn
                -b' Zu  nn'  r
                Za  Zb   a   b
                Zb' Zu  Zb' Zu];
            phi = expm(M*this.Ts);
            phi12 = phi(1:n,n+1:2*n);
            phi22 = phi(n+1:2*n,n+1:2*n);
            QQ = phi22'*phi12;
            QQ = (QQ+QQ')/2;        % Make sure QQ is symmetric
            qd = QQ(1:Nx,1:Nx);
            rd = QQ(Nx+1:n,Nx+1:n);
            nd = QQ(1:Nx,Nx+1:n);
            ad = phi22(1:Nx,1:Nx);
            bd = phi22(1:Nx,Nx+1:n);
            
            this.Rd = rd;
            this.Qd = qd;
            
            this.Ad = ad;
            this.Bd = bd;
            this.Nd = nd;
            this.Cd = eye(2);
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
            updatePerformanceWeights(this);
            
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
        
        function set.PlotValue(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','PlotValue');
            this.PlotValue = val;
        end
        function set.A(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','V0y');
            this.A = val;
        end
        function set.Q(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[2 2]},'','Q');
            this.Q = val;
            updatePerformanceWeights(this);
        end
        function set.R(this,val)
            validateattributes(val,{'numeric'},{'real','finite','size',[1 1]},'','R');
            this.R = val;
            updatePerformanceWeights(this);
        end
        function set.Gain(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','Gain');
            this.Gain = val;
            updatePerformanceWeights(this);
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
            updatePerformanceWeights(this);
        end
        
        function resetEnv(this)
            %             this.State = [0,0];
            this.V0y = 0;
        end
        
        function [nextobs,rwd,isTerminal] = step(this,s,action,plotValue)
            this.PlotValue = plotValue;
            %Action: we have to conside the presence of force of gravity
            
            action =this.act+ getForce(this,action)
            this.act = action;
            % Time Step
            %             t = this.Ts;
            % Updating Enviroment State
            x = s.';
            xk1 = this.Ad*x+this.Bd*action;
            % Saturating position and velocity
            xk1(1) = min(max(xk1(1),this.lby),this.uby);
            
            xk1(2) = min(max(xk1(2),this.lbv),this.ubv);
            % Output Linearized System y(t)= [pos;vel] = C*x
            nextobs = this.Cd*xk1;
            this.State = xk1;
            
            %             this.State = s;
            %Get next state
            % We have to describe the elevator dynamics along x-axis and
            % y-axis.
            % x-axis: for each time step x(t) = x0
            % y-axis: for each time step y(t) = y0+v0y*t+1/2*ay*(t)^2
            %             yt = this.State(1);
            %             v0t = this.State(2);
            %             at = this.A+action;
            %             atp = at - this.G;
            %             ytp = yt+v0t*t+0.5*atp*(t)^2;
            %             if ytp < this.lby || ytp >= this.uby
            %                 this.resetEnv()
            %                 this.A = 9.8;
            %                 rwd = -1;
            %                 ytp = min(max(ytp,this.lby),this.uby);
            %                 sp = [ytp;0];
            %                 this.State = sp;
            %                 isTerminal = false;
            %                 return
            %             end
            %
            %             ytp = min(max(ytp,this.lby),this.uby);
            %             this.V0y = min(max(v0t+atp*t,this.lbv),this.ubv);
            %
            %             sp = [ytp;this.V0y];
            % %             if norm(sp) < this.GoalThreshold
            % % %                 this.resetEnv()
            % % %                 this.A = 9.8;
            % %                 rwd = -1;
            % %                 sp = [0;0];
            % %                 this.State = sp;
            % %                 isTerminal = false;
            % %                 return
            % %             end
            %             this.State = sp;
            %
            %             this.A = at;
            %             The episode will terminate under the following conditions:
            %             1. the mass moves more than X units away from the origin
            %             2. the norm of the state is less than some threshold
            %
            %             The second point is critical for training as it prevents the
            %             replay buffer being saturated with 0s for training
            isdone = nextobs == this.FinalPoint ;%|| norm(sp) < this.GoalThreshold ;
            if nextobs == this.lby
                this.act = 0;
                xk1(2) = 0;
                nextobs(2) = 0;
                rwd = -100;
            else
%                 rwd = - x'*this.Qd*x - action'*this.Rd*action - 2*x'*this.Nd*action;
                rwd = -1;
            end
            if isdone == 1
                isTerminal = true;
            else
                isTerminal = false;
            end
            
        end
        
    end
end
