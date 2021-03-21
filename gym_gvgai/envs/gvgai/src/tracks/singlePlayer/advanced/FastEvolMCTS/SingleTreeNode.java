package tracks.singlePlayer.advanced.FastEvolMCTS;

import core.game.Observation;
import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Utils;
import tools.Vector2d;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
//NOT SURE IF ALLOWED

public class SingleTreeNode
{
    //evolutionary strategy attempt: euclidean distances from the avatar to the different types of closest NPC, resource, non-static object and portal.
    public static Random r = new Random();
    public static boolean initialised = false;
    public static double[] weights = new double[10];
    public static double[] new_weights = new double[10];
    public static double performance_weights = 0;
    public static double performance_new_weights = 0;
    public static double[] sigmas = new double[10];
    public static double tau_0 = 1/(Math.sqrt(2*sigmas.length));
    public static double tau_1 = 1/(Math.sqrt(2*Math.sqrt(sigmas.length)));

    private final double HUGE_NEGATIVE = -10000000.0;
    private final double HUGE_POSITIVE =  10000000.0;
    public double epsilon = 1e-6;
    public double egreedyEpsilon = 0.05;
    public SingleTreeNode parent;
    public SingleTreeNode[] children;
    public double totValue;
    public int nVisits;
    public Random m_rnd;
    public int m_depth;
    protected double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    public int childIdx;

    public int num_actions;
    Types.ACTIONS[] actions;
    public int ROLLOUT_DEPTH = 6;//10
    public double K = Math.sqrt(2);

    public StateObservation rootState;

    public SingleTreeNode(Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(null, -1, rnd, num_actions, actions);
        if(!initialised){
            for(int i = 0; i< weights.length; i++) {
                weights[i] = r.nextDouble();
                sigmas[i] = 0.25;
            }
            initialised= true;
        }
    }

    public SingleTreeNode(SingleTreeNode parent, int childIdx, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNode[num_actions];
        totValue = 0.0;
        this.childIdx = childIdx;
        if(parent != null)
            m_depth = parent.m_depth+1;
        else
            m_depth = 0;
    }


    public void mctsSearch(ElapsedCpuTimer elapsedTimer) {
        // mutate sigmas and weights
        double mut_0 = r.nextGaussian()*tau_0;
        for(int i = 0; i < sigmas.length;i++){
            double mut_1 = r.nextGaussian()*tau_1;
            sigmas[i] = sigmas[i] * Math.exp(mut_0+mut_1);
            new_weights[i] = weights[i] + r.nextGaussian()*sigmas[i];
            if(new_weights[i] > 5) new_weights[i] = 5;
            if(new_weights[i] < -5) new_weights[i] = -5;
        }


        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = elapsedTimer.remainingTimeMillis();
        int numIters = 0;

        int remainingLimit = 5;

        //WHILE TIME LEFT
        while(remaining > 2*avgTimeTaken && remaining > remainingLimit){
        //while(numIters < Agent.MCTS_ITERATIONS){

            StateObservation state = rootState.copy();

            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy(state);
            double delta = selected.rollOut(state);
            backUp(selected, delta);

            numIters++;
            acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
            //System.out.println(elapsedTimerIteration.elapsedMillis() + " --> " + acumTimeTaken + " (" + remaining + ")");
            avgTimeTaken  = acumTimeTaken/numIters;
            remaining = elapsedTimer.remainingTimeMillis();
        }
        //selection: is new_weights better than weights?
        System.out.println("comparison " +performance_weights + " " + performance_new_weights);
        if (performance_new_weights >=performance_weights){
            weights = new_weights.clone(); // not sure if i need a deep copy here
            performance_weights = performance_new_weights;
        }
        System.out.println("WEIGHTS" + Arrays.toString(weights) + " performance " + performance_weights);
    }

    public SingleTreeNode treePolicy(StateObservation state) {

        SingleTreeNode cur = this;
        while (!state.isGameOver() && cur.m_depth < ROLLOUT_DEPTH)
        {
            if (cur.notFullyExpanded()) {

                return cur.expand(state);

            } else {
                cur = cur.uct(state);
            }
        }

        return cur;
    }


    public SingleTreeNode expand(StateObservation state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        state.advance(actions[bestAction]);
        SingleTreeNode tn = new SingleTreeNode(this,bestAction,this.m_rnd,num_actions, actions);
        children[bestAction] = tn;
        return tn;
    }

    public SingleTreeNode uct(StateObservation state) {

        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + this.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);
            //System.out.println("norm child value: " + childValue);

            double uctValue = childValue +
                    K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + this.epsilon));

            uctValue = Utils.noise(uctValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
            + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        state.advance(actions[selected.childIdx]);

        return selected;
    }

    public static ArrayList<double[]> events = new ArrayList<>();
    public double rollOut(StateObservation state)
    {

        //initialise arraylists to hold items needed to determine collisions (/events)
        ArrayList<double[]> distances = new ArrayList<>();
        ArrayList<Observation> collidables = new ArrayList<>();
        ArrayList<Observation> avatar_produced = new ArrayList<>();
        int avatar_type = state.getAvatarType();

        // initialise the depth, and save the value of the state before the rollout
        int thisDepth = this.m_depth;
        double current_val = value(state);
        while (!finishRollout(state,thisDepth)) {
            /**
            For the Knowledge items
            1. get collision points NPC, Movable, immovable etc etc
            2. get avatar and avatar sprites position
            3. check for collisions
            4. if there is a collision go through arraylist to see if it is in it, if not, then add new entry
             */

            current_val = value(state);
            //for fast evolutionary, decide action based on linear weighted some of distances
            double[] action_values = new double[num_actions];
            for(int k = 0; k< num_actions;k++){
                StateObservation st = state.copy();
                st.advance(actions[k]);
                // EVALUATE IE GET DISTANCES
                collidables = new ArrayList<>();
                if (!Objects.isNull(st.getNPCPositions())) { for(int i = 0; i <  st.getNPCPositions().length; i ++) collidables.addAll(st.getNPCPositions()[i]); }
                if (!Objects.isNull(st.getImmovablePositions())) { for(int i = 0; i <  st.getImmovablePositions().length; i ++) collidables.addAll(st.getImmovablePositions()[i]); }
                if (!Objects.isNull(st.getMovablePositions())){ for(int i = 0; i <  st.getMovablePositions().length; i ++) collidables.addAll(st.getMovablePositions()[i]); }
                if (!Objects.isNull(st.getResourcesPositions())){ for(int i = 0; i <  st.getResourcesPositions().length; i ++) collidables.addAll(st.getResourcesPositions()[i]); }
                if (!Objects.isNull(st.getPortalsPositions())) { for(int i = 0; i <  st.getPortalsPositions().length; i ++) collidables.addAll(st.getPortalsPositions()[i]); }
                // Get avatar thingd
                avatar_produced = new ArrayList<>();
                if (!Objects.isNull(st.getFromAvatarSpritesPositions())) avatar_produced.addAll(st.getFromAvatarSpritesPositions()[0]);
                Vector2d avatar_pos = st.getAvatarPosition();

                ArrayList<double[]>dist = get_start_distances(collidables, avatar_produced, avatar_pos, avatar_type);
                for(int j = 0; j< dist.size(); j++) {
                    if (j < new_weights.length)
                        action_values[k] += new_weights[j] * dist.get(j)[2];
                }
            }
            // evaluate which action to take with a softmax
            double[] exp_action_values = new double[num_actions];
            double exp_action_sum = 0;
            for(int i =0 ; i< exp_action_values.length; i++){
                double temp = Math.exp(-action_values[i]);
                exp_action_values[i] = temp;
                exp_action_sum += temp;
            }
            int best = 0; double best_val = 0;
            for(int i = 0; i < exp_action_values.length; i++){
                double temp = exp_action_values[i]/exp_action_sum;
                if (temp > best_val){
                    best_val = temp;
                    best = i;
                }
            }
            state.advance(actions[best]);
            thisDepth++;

        }

        double delta = value(state);

        if(delta < bounds[0])
            bounds[0] = delta;
        if(delta> bounds[1])
            bounds[1] =delta;

        //double normDelta = utils.normalise(delta ,lastBounds[0], lastBounds[1]);
        //System.out.println("VALUE" + (value+delta));
        performance_new_weights = delta;
        return delta;
    }

    public ArrayList<double[]> get_start_distances(ArrayList<Observation> sprites, ArrayList<Observation> avatar_sprites, Vector2d avatar_pos, int avatar_type){
        /**
         * Calculates the shortest distance for each type of avatar / avator produced sprite to each type of collidables
         */
        ArrayList<double[]> distances = new ArrayList<>();
        for(Observation sprite: sprites){
            for (Observation avatar_sprite: avatar_sprites) {
                boolean found = false;
                for (double[] d : distances) {
                    if (d[0] == avatar_sprite.itype && d[1] == sprite.itype && avatar_sprite.position.dist(sprite.position) < d[2]) {
                        d[2] = avatar_sprite.position.dist(sprite.position);
                        found = true;
                        break;
                    }
                    if(d[0] == avatar_sprite.itype && d[1] == sprite.itype) found = true;
                }
                if(!found) {
                    double[] temp = {avatar_sprite.itype, sprite.itype, avatar_sprite.position.dist(sprite.position),-1};
                    distances.add(temp);
                }
            }
            boolean found = false;
            for (double[] d : distances) {
                if (d[0] == avatar_type && d[1] == sprite.itype && avatar_pos.dist(sprite.position) < d[2]) {
                    d[2] = avatar_pos.dist(sprite.position);
                    found = true;
                    break;
                }
                if (d[0] == avatar_type && d[1] == sprite.itype) found = true;
            }
            if(!found) {
                double[] temp = {avatar_type, sprite.itype, avatar_pos.dist(sprite.position),-1};
                distances.add(temp);
            }
        }
        return distances;
    }

    public double value(StateObservation a_gameState) {

        boolean gameOver = a_gameState.isGameOver();
        Types.WINNER win = a_gameState.getGameWinner();
        double rawScore = a_gameState.getGameScore();

        if(gameOver && win == Types.WINNER.PLAYER_LOSES)
            rawScore += HUGE_NEGATIVE;

        if(gameOver && win == Types.WINNER.PLAYER_WINS)
            rawScore += HUGE_POSITIVE;

        return rawScore;
    }

    public boolean finishRollout(StateObservation rollerState, int depth)
    {
        if(depth >= ROLLOUT_DEPTH)      //rollout end condition.
            return true;

        if(rollerState.isGameOver())               //end of game
            return true;

        return false;
    }

    public void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }


    public int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null)
            {
                if(first == -1)
                    first = children[i].nVisits;
                else if(first != children[i].nVisits)
                {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = Utils.noise(childValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }
        return selected;
    }

    public int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                //double tieBreaker = m_rnd.nextDouble() * epsilon;
                double childValue = children[i].totValue / (children[i].nVisits + this.epsilon);
                childValue = Utils.noise(childValue, this.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    public boolean notFullyExpanded() {
        for (SingleTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
