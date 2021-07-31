% Elevator - Reinforcment Learning

% The reinforcement learning problem is getting an optimal policy for controlling
% an elevator. In particular, the actions are represented by the possible increases
% of the forces that allow the movement of the lift, which for simplicity has been
% considered to have a unitary mass, while the states are all the possible positions
% that the lift can assume in space. Given the high cardinality of the states and
% the continuous nature of the problem, we have chosen to solve this problem with the
% application of the SARSA_RBF typical of the functional approximation.
clc
clear
close all

%% Init

% Initial Point

yStart = 0;
% Constant of increments

K=0.3;
% Action List

action = K*[-1 0 1];
% Lower and UpperBound to y position

lby = 0;
uby = 8;
% Lower and UpperBound to velocity

lbv = -2;
ubv = 8;
% Grid Dimension

M =10;
% Number of grid

N = 1;
% Number of episodes

numEpisodes = 1e5;
% RBF params

epsilon = 2e-1;
alpha = 1e-3;
gamma = 1;
sigma = 0.01;
% Number of RBF Cells

nCells = (M+1)^2;
d = length(action)*N*nCells;
% Costruction of tiles
% To build tiles we need to bound y position between intial and final
% point. Due to the nature of the problem we have to consider bounded
% velocity to safety of the people in the elevator

[gridx, gridv] = build_tiles(lby, uby,lbv,ubv, M, N);

% Init Elevatr Enviroment
env = ElevatorConcrete;

%Plotting Enviroment
% env.plot;

% History of all episodes

%% TRAINING PHASE - IMPLEMENTING SARSA RBF ALGORITHM
w = zeros(d,1);
% Last Episode is an array of type [r,y,a]
LastEpisode =  [];
for ii = 1:numEpisodes
    s = env.State;
    a = epsgreedy(s, w, epsilon, gridx, gridv, M, N, action);
    isTerminal = 0;
    steps = 0;
    while ~isTerminal
        
        steps = steps + 1;
        x = getRBF(s, a, sigma, gridx, gridv, M, N, action);
        [sp, r, isTerminal] = env.step(s,action(a),0);
        if ii == numEpisodes
            LastEpisode(end+1,:) = [r,s(1),s(2),a];
        end
        if isTerminal
            w = w + alpha*(r - w'*x)*x;
        else
            ap = epsgreedy(sp, w, epsilon, gridx, gridv, M, N, action);
            xp = getRBF(s, a, sigma, gridx, gridv, M, N, action);
            w = w + alpha*(r + gamma*w'*xp - w'*x)*x;
        end
        s = sp;
        a = ap;
        
        
        %         if ii == numEpisodes
        %             clf
        %             pxx = linspace(lbx, ubx);
        %             plot(pxx, sin(3*pxx),'k','linewidth',2);
        %             hold on
        %             plot(s(1), sin(3*s(1)),'b*');
        %             title([' Action = ', num2str(a),...
        %                 ' - Episode = ',num2str(ii),...
        %                 ' - Steps = ',num2str(steps')])
        %             pause(0.1)
        %         end
        
    end
    
    disp([ii, steps])
    env.resetEnv();
    if ii ==numEpisodes
        isTerminal = false;
        s = [LastEpisode(1,2);LastEpisode(1,3)];
        i=1;
        while ~isTerminal
            [sp,rwd,isTerminal] = env.step(s,action(LastEpisode(i,4)),1);
            env.plot
            pause()
            s = sp
            i=i+1;
        end
    end
end

save ElevatorEpisode LastEpisode
%%
% while env.R
% % [rwd,isdone,loggedSignals]=env.step(3);
% a ~= action(randi(size(action,2)))
% [rwd,isdone,loggedSignals]=env.step(a);
% env.plot;
% env.A
% env.V0y
% pause()
% end
clc
close all

load ElevatorEpisode
% Init Elevatr Enviroment
env = ElevatorConcrete;

%Plotting Enviroment
env.plot;
%% SIMULATING LAST EPISODE


