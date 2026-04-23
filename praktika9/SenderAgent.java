import jade.core.Agent;
import jade.core.AID;
import jade.core.behaviours.OneShotBehaviour;
import jade.lang.acl.ACLMessage;

public class SenderAgent extends Agent {

    @Override
    protected void setup() {
        System.out.println(getLocalName() + " started.");

        addBehaviour(new OneShotBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
                msg.addReceiver(new AID("receiver", AID.ISLOCALNAME));
                msg.setContent("Hello from SenderAgent!");
                send(msg);

                System.out.println(getLocalName() + " sent: " + msg.getContent());
            }
        });
    }
}