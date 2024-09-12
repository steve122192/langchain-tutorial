from chains_graph import retrieval_chain, agent_executor, question_router

####################################
# Define State
####################################
user_input = 'What is the patients name?'

state_dict = {'input': user_input,
              'router_response': None,
              'answer': None,
              'documents': []}

####################################
# Define Nodes
####################################
def route(state_dict):
    response = question_router.invoke(input={"question": state_dict["input"]})
    return response.datasource

def retrieve(state_dict):
    response = retrieval_chain.invoke(input={"input": state_dict["input"]})
    return (response['answer'], [(doc.metadata['source'], doc.metadata['page']) for doc in response['context']])

def search(state_dict):
    response = agent_executor.invoke(input={"input": state_dict["input"]})
    return response['output']

    
####################################
# Build Graph
####################################  
answer =  False
while not answer:
    # check if route has been selected
    if not state_dict['router_response']:
        state_dict['router_response'] = route(state_dict)
     # check if answer has been given   
    elif not state_dict['answer']:
        # check selected route
        if state_dict['router_response'] == 'vectorstore':
            state_dict['answer'] = retrieve(state_dict)[0]
            state_dict['documents'] = retrieve(state_dict)[1]
        elif state_dict['router_response'] == 'websearch':
            state_dict['answer'] = search(state_dict)
    # end if answer has been givem
    elif state_dict['answer']:
        answer = True
        print(state_dict['answer'])
        if state_dict['router_response'] == 'vectorstore':
            print(state_dict['documents'])

