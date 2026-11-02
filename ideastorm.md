### Goal :
to build a learning system where it's problem solving and self discovery learning first
we are building GRAPHY and it's companion

## **GRAPHY**:
- it's a system I throw in a textbook It splits into chunks and find embedding similarly for each chunk among all the other chunk and put that as a weighted edge between each nodes which is the chunks I will use a small LLM to see whether I need chunk A for chunk B
- If A is prerequisites of B then edge go from A to B
- Each chunk or node is given a score which is sum of outgoing node weight minus sum of incoming node weight The chunk with highest score is the fundamental topic which needs to be studied first Then I take a problem set for the book either from the book itself of external source Then I run similarity check with each chunk and get a score for each chunk for that problem Then I multiply that score for each chunk for that problem with the foundation score of each chunks
- The final number that we get for each problem is the problems foundationess
- So it's good to start with problems of highest foundation score
- Then I choose set of four problem from the top of the list and it's each chunk details Get the chunks data pass it to LLM and ask it to give me the bare minimum topics and concepts needed and rest of the things should be self discovered by me
- 
- Then I will go on to solve the problems 
- All the solutions to the problems would be analysed and then determine whether I self discovered the topics If not then LLM would tell me those topics along with the bare bone concepts for next set of problems If I failed solving the problems Then LLM not only give the un discovered topics but also the topics that will help me to not struggle.

### Constraints for Making GRAPHY
- strictly use containerization through podman and get the container details like files and images to make it after the completion of this project
- I have installed podman in my **Arch Linux** machine through pacman but if you want any dependancy fix or any fix related to installation, feel free to do so
- use git to track our project and name that repo as 'GRAPHY'
- ignore extremely big files in git so i can push it to my Github but do make sure it can be remade, eg requirements.txt instead of entire venv directry 
- YOU ARE NOT ALLOWED TO PUSH TO MY GITHUB
- JUST TRACK EVERYTHING IN GIT, I WILL PUSH IT MYSELF
- we are mostly gonna deal with Math, Physics, Computer Science textbook so make sure the chunking and embedding for these complex books are perfect, if you feel existing tools are problem then fix it yourself and rememer to focus on general scope instead of specific book. remember i should not have any problem with chunking and embedding for any book
- you can extract problem from the book itself but i will also give extra problem set (course assignment, exams etc) but make sure to make a automated pipeline to extract those problems
- you may use Local AI for any of the pipeline but make sure to use the capable one and use Agent Harnessing to maximize the potential
- and make sure embedding is accurate, you may use Local AI models for this embedding task but just make sure it can understand code, figures and technical, scientific and academic terms
- you may use special data types for graph like GraphAr or any better one, because we are dealing with big directed and weighted graph
- to store score for each node/chunk you may use JSON
- use JSON to store both Chunk's score for each problem 
- also store overal net foundationess score of a problem in JSON
- use Local AI to determine the pre request of each chunk
- and also use Local AI for determining the bare minimum concepts needed, analyzing solutions, giving the rest of self discovery topic and topic for next set of problems
- this is my Machine's specs:
=== CPU ===
CPU(s):                                  12
Model name:                              13th Gen Intel(R) Core(TM) i5-1335U

=== RAM ===
Mem:            15Gi       6.5Gi       838Mi       1.1Gi       8.6Gi       8.8Gi

=== GPU ===
00:02.0 VGA compatible controller: Intel Corporation Raptor Lake-P [Iris Xe Graphics] (rev 04)

- sure it may be weak but you can optimize my system specs to it's core and unapologitically squeeze out every bit of performance from my machine
- and you can use Vector database, RAG and Agent Harnessing to split the task into small task and let the agents focus on small problems
- i guess this should make Local AI more powerful
- you are allowed to use suitable AI model for it and any framework but keep our performance demand in mind

### How the final thing should be
- i don't care if it's a podman container that i can spin up and use
- or a neat GUI app or a local web instance
- I'm technical enough to read the output from terminal also
- the main point is do not change our system or tradeoff anything in it for neat frontend
- but do MAKE SURE IT'S USABLE ATLEASE
- finally document the entire thing along with instruction to use
- it should be reliable and work everytime 
- don't let the bugs creep in and give bad result
- REMEMBER PEOPLE ARE GONNA USE THIS TO LEARN SO BE RESPONSIBLE 

- you may ask for any clarifications if you want along the way