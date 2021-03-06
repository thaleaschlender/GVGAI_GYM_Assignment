package tracks.singlePlayer.advanced.KBMCTS;

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
    public static Random r = new Random();

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
    public int ROLLOUT_DEPTH = 8;//10
    public double K = Math.sqrt(2);

    public StateObservation rootState;

    public SingleTreeNode(Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(null, -1, rnd, num_actions, actions);
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
    public double rollOut(StateObservation state) {
        //deepcopy the events
        ArrayList<double[]> start = new ArrayList<>();
        for (double[] doubles : events) start.add(doubles.clone());

        //initialise arraylists to hold items needed to determine collisions (/events)
        ArrayList<double[]> distances = new ArrayList<>();
        ArrayList<Observation> collidables = new ArrayList<>();
        ArrayList<Observation> avatar_produced = new ArrayList<>();
        int avatar_type = state.getAvatarType();

        // initialise the depth, and save the value of the state before the rollout
        int thisDepth = this.m_depth;
        double current_val = value(state);
        while (!finishRollout(state, thisDepth)) {
            /**
             For the Knowledge items
             1. get collision points NPC, Movable, immovable etc etc
             2. get avatar and avatar sprites position
             3. check for collisions
             4. if there is a collision go through arraylist to see if it is in it, if not, then add new entry
             */
            collidables = new ArrayList<>();
            if (!Objects.isNull(state.getNPCPositions())) {
                for (int i = 0; i < state.getNPCPositions().length; i++) collidables.addAll(state.getNPCPositions()[i]);
            }
            if (!Objects.isNull(state.getImmovablePositions())) {
                for (int i = 0; i < state.getImmovablePositions().length; i++)
                    collidables.addAll(state.getImmovablePositions()[i]);
            }
            if (!Objects.isNull(state.getMovablePositions())) {
                for (int i = 0; i < state.getMovablePositions().length; i++)
                    collidables.addAll(state.getMovablePositions()[i]);
            }
            if (!Objects.isNull(state.getResourcesPositions())) {
                for (int i = 0; i < state.getResourcesPositions().length; i++)
                    collidables.addAll(state.getResourcesPositions()[i]);
            }
            if (!Objects.isNull(state.getPortalsPositions())) {
                for (int i = 0; i < state.getPortalsPositions().length; i++)
                    collidables.addAll(state.getPortalsPositions()[i]);
            }
            // Get avatar thingd
            avatar_produced = new ArrayList<>();
            if (!Objects.isNull(state.getFromAvatarSpritesPositions()))
                avatar_produced.addAll(state.getFromAvatarSpritesPositions()[0]);
            Vector2d avatar_pos = state.getAvatarPosition();
            boolean calc_distance = false; // calculate inital distances for KB
            if (thisDepth == this.m_depth) calc_distance = true;
            //collisions
            for (Observation collidable : collidables) {
                for (Observation avatar_made : avatar_produced) {
                    if (collidable.position.dist(avatar_made.position) < state.getBlockSize()) {
                        //check for events
                        int a = avatar_made.itype;
                        int b = collidable.itype;
                        boolean found = false;
                        for (double[] event : events) {
                            if (event[0] == a && event[1] == b) {
                                event[2] += 1;
                                event[3] += (value(state) - current_val);
                                event[3] = event[3] / 2;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            double[] temp = {a, b, 1, (value(state) - current_val)};
                            events.add(temp);
                        }
                    }
                    if (calc_distance) {
                        boolean found = false;
                        int a = avatar_made.itype;
                        int b = collidable.itype;
                        for (double[] d : distances) {
                            if (d[0] == a && d[1] == b && avatar_made.position.dist(collidable.position) < d[2]) {
                                d[2] = avatar_made.position.dist(collidable.position);
                                found = true;
                                break;
                            }
                            if (d[0] == avatar_made.itype && d[1] == collidable.itype) found = true;
                        }
                        if (!found) {
                            double[] temp = {avatar_made.itype, collidable.itype, avatar_made.position.dist(collidable.position), -1};
                            distances.add(temp);
                        }
                    }
                }
                if (collidable.position.dist(avatar_pos) < state.getBlockSize()) {
                    //check for events
                    int b = collidable.itype;
                    boolean found = false;
                    for (double[] event : events) {
                        if (event[0] == avatar_type && event[1] == b) {
                            event[2] += 1;
                            event[3] += (value(state) - current_val);
                            event[3] = event[3] / 2;
                            event[3] = Math.round(event[3] * 10000.0) / 10000.0;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        double[] temp = {avatar_type, b, 1, (value(state) - current_val)};
                        temp[3] = Math.round(temp[3] * 10000.0) / 10000.0;
                        events.add(temp);
                    }
                }
                if (calc_distance) {
                    boolean found = false;
                    for (double[] d : distances) {
                        if (d[0] == avatar_type && d[1] == collidable.itype && avatar_pos.dist(collidable.position) < d[2]) {
                            d[2] = avatar_pos.dist(collidable.position);
                            found = true;
                            break;
                        }
                        if (d[0] == avatar_type && d[1] == collidable.itype) found = true;
                    }
                    if (!found) {
                        double[] temp = {avatar_type, collidable.itype, avatar_pos.dist(collidable.position), -1};
                        distances.add(temp);
                    }
                }
            }
            current_val = value(state);


        int action = m_rnd.nextInt(num_actions);
        state.advance(actions[action]);
        thisDepth++;

    }

        double delta = value(state);
        double value = delta - current_val;
        if (value == 0){
            double curiosity = curiosity(start);
            double distance = distance(collidables,avatar_produced, state.getAvatarPosition(), avatar_type, distances);
            value = 0.66 * curiosity + 0.33  * distance;
        }
        if(value < bounds[0])
            bounds[0] = value;
        if(value > bounds[1])
            bounds[1] =value;

        //double normDelta = utils.normalise(delta ,lastBounds[0], lastBounds[1]);
        //System.out.println("VALUE" + (value+delta));
        return value +delta;
        //return delta;
    }
    public double curiosity(ArrayList<double[]> old_events){
        double curiosity = 0;
        for (double[] event: events) {
            boolean found = false;
            for(double[] old_event: old_events){
                if (event[0] == old_event[0] && event[1] == old_event[1]){
                    curiosity += (event[2]/old_event[2]) -1;
                    found = true;
                }
            }
            if (!found) {
                curiosity += event[2];
            }
        }
        return curiosity;
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
    public double distance(ArrayList<Observation> sprites, ArrayList<Observation> avatar_sprites, Vector2d avatar_pos, int avatar_type, ArrayList<double[]> distances){
        /**
         * Calculates the distance Knowledge Base parameter at the end of the rollout.
         **/
        double distance = 0;
        // COMPUTE DISTANCES
        for (Observation avatar_sprite: avatar_sprites) {
            for(Observation sprite: sprites){
                for (double[] d : distances) {
                    if (d[0] == avatar_sprite.itype && d[1] == sprite.itype && (d[3]== -1 || avatar_sprite.position.dist(sprite.position) < d[3] ) ){
                        d[3] = avatar_sprite.position.dist(sprite.position);
                        break;
                    }
                    if (d[0] == avatar_type && d[1] == sprite.itype && (d[3] == -1 || avatar_pos.dist(sprite.position) < d[3])) d[3] = avatar_pos.dist(sprite.position);
                }
            }
        }

        // loop through events
        for (double[] d: distances) {
            boolean found = false;
            for(double[] event: events){
                if (d[0] == event[0] && d[1] == event [1]){
                    if(event[3] > 0&& d[2] > 0) {
                        if (d[3] == -1) d[3] =0;
                        //System.out.println("distance for something"+ (1 - (d[3]/d[2])));
                        distance += (1 - (d[3]/d[2]));
                        found = true;
                        break;
                    }
                }
            }
            if(!found && d[2]>0){
                if (d[3] == -1) d[3] = 0;
                //System.out.println("distance for something"+ (1 - (d[3]/d[2])));
                distance += (1 - (d[3]/d[2]));
            }

        }
        return distance;

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
