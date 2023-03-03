**TASK 1 PROBLEM STATEMENT**

Building on the minimax project 

1. The game consists of an NxN chessboard and N pawns for each player. White starts, as always. (N variable)

2. Pawns can either capture another pawn of the opponents with a single square diagonal movement or advance 1 square        (basic chess rules of a pawn). No En Passant or initial 2 square advances allowed. 

3. The terminal state is when the player whose turn it has no allowed movements (all current pawns are blocked by      another pawn, can’t capture, or reached the other end). The winner is decided by who has more pawns at the opponent's   end. If both have the same number, it’s a draw.

4. Implement the ‘AI’ of the game using the Minimax algorithm, and improve its performance in some explainable way using    any additional techniques of your choice.

5. Brownie points:  
   Make the max depth of the tree variable up to the user who is playing the game as a variation of difficulty.
   

**SCREENSHOTS OF TASK 1 OUTPUT**

TESTCASE 1: N = 7, DEPTH = 5, user = WHITE
<img width="821" alt="image" src="https://user-images.githubusercontent.com/62715046/221759412-0845dfa6-d1b3-47cc-b980-8d50346de233.png">

<img width="457" alt="image" src="https://user-images.githubusercontent.com/62715046/221759483-10415abc-b8b9-414a-b635-81d7adce5d2b.png">

<img width="378" alt="image" src="https://user-images.githubusercontent.com/62715046/221759532-bf2b9cf4-3fb5-4ff0-bffc-0e690a820600.png">

<img width="380" alt="image" src="https://user-images.githubusercontent.com/62715046/221759559-e84e4854-e2cd-4978-94ce-036d6ecdb916.png">

<img width="382" alt="image" src="https://user-images.githubusercontent.com/62715046/221759596-8f1049d7-0fc4-4ae1-afed-f706a4199e42.png">

<img width="387" alt="image" src="https://user-images.githubusercontent.com/62715046/221759624-36727a1d-5978-4072-8a7e-e365b17c3bb6.png">

<img width="380" alt="image" src="https://user-images.githubusercontent.com/62715046/221759657-b3104a79-146f-4ba0-85e8-3d616a41fc66.png">

<img width="383" alt="image" src="https://user-images.githubusercontent.com/62715046/221759690-28bf30ea-985b-4897-b4d4-641813ffea0e.png">

<img width="380" alt="image" src="https://user-images.githubusercontent.com/62715046/221759719-454b457c-4c82-4a24-884a-8ee67db51059.png">

<img width="381" alt="image" src="https://user-images.githubusercontent.com/62715046/221759762-fd0077eb-ffef-4dfd-b595-a4fcad28f925.png">

<img width="379" alt="image" src="https://user-images.githubusercontent.com/62715046/221759797-c51f7a51-7c79-4a98-abab-b31714c733b3.png">

<img width="383" alt="image" src="https://user-images.githubusercontent.com/62715046/221759821-3cec677a-46af-4f7c-af46-6af99132768b.png">

<img width="392" alt="image" src="https://user-images.githubusercontent.com/62715046/221759847-2da0ef09-8e6d-41ef-b266-0dd1eecbe8c5.png">

<img width="396" alt="image" src="https://user-images.githubusercontent.com/62715046/221760136-099bf0f1-c4ee-4f72-bdac-0ea11d9c07b1.png">

<img width="400" alt="image" src="https://user-images.githubusercontent.com/62715046/221760180-ea204834-35e3-4dc5-923d-e448f8ed7473.png">

<img width="383" alt="image" src="https://user-images.githubusercontent.com/62715046/221760203-0415df4a-7a01-48a6-8473-a813698487a9.png">

<img width="395" alt="image" src="https://user-images.githubusercontent.com/62715046/221760229-7b315f6d-8f4a-45a5-b95f-fb8d0f2f0d24.png">

<img width="389" alt="image" src="https://user-images.githubusercontent.com/62715046/221760261-3d3ac6a4-09be-4d4b-a568-5f08b59980b8.png">

<img width="391" alt="image" src="https://user-images.githubusercontent.com/62715046/221760290-0d99b033-0087-478b-9737-1dcadf31a582.png">

<img width="396" alt="image" src="https://user-images.githubusercontent.com/62715046/221760342-2293b640-23f6-4344-bb8e-5493cf695cc1.png">

<img width="382" alt="image" src="https://user-images.githubusercontent.com/62715046/221760367-81078b0e-d2e9-4d3d-b6bc-6e53b4d37681.png">

<img width="385" alt="image" src="https://user-images.githubusercontent.com/62715046/221760401-952df077-a01a-431d-8560-42202bdafade.png">

<img width="384" alt="image" src="https://user-images.githubusercontent.com/62715046/221760428-0601bc9c-db40-4f3c-94d7-c7aa5bf60936.png">

<img width="297" alt="image" src="https://user-images.githubusercontent.com/62715046/221760456-e6e4d542-63ff-49bb-bf64-bd293c8d646f.png">


TESTCASE 2: N = 5, DEPTH = 7, user = BLACK
<img width="831" alt="image" src="https://user-images.githubusercontent.com/62715046/221774031-286afb73-e692-4ade-baae-1ff7b1915ffa.png">

<img width="268" alt="image" src="https://user-images.githubusercontent.com/62715046/221774079-2d32088a-4792-4f99-b490-2efe63534df0.png">

<img width="380" alt="image" src="https://user-images.githubusercontent.com/62715046/221774111-27b9e2ef-eb1d-445d-95b6-24fb8987abfa.png">

<img width="396" alt="image" src="https://user-images.githubusercontent.com/62715046/221774156-71253313-8209-4fb7-9106-7dccc270e2bd.png">

<img width="402" alt="image" src="https://user-images.githubusercontent.com/62715046/221774202-f4bfc799-7a9a-4b1b-8973-7f5c06a45962.png">

<img width="385" alt="image" src="https://user-images.githubusercontent.com/62715046/221774241-e594a73a-dad6-4643-92fe-443a01006f34.png">

<img width="394" alt="image" src="https://user-images.githubusercontent.com/62715046/221774274-1dd8fc47-1c71-4336-a4a7-272b71edde33.png">

<img width="391" alt="image" src="https://user-images.githubusercontent.com/62715046/221774346-b2e077b6-220e-42d4-b79b-38a7ce278c79.png">

<img width="376" alt="image" src="https://user-images.githubusercontent.com/62715046/221774404-65b89cf0-d89e-4c42-ab58-407ba54cb922.png">

<img width="389" alt="image" src="https://user-images.githubusercontent.com/62715046/221774450-8dbe2553-8018-442e-a9c0-3d0efa834e4a.png">

<img width="376" alt="image" src="https://user-images.githubusercontent.com/62715046/221774503-e2aa1849-7606-4199-895f-e79d1ee8e94d.png">

<img width="300" alt="image" src="https://user-images.githubusercontent.com/62715046/221774555-35388ac1-57ce-4b55-a7e6-6e76a0570a44.png">
