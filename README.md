# NanoGPTExperiment
My attempt at building a generative pretrained transformer ML model, like Chat GPT.
My primary goal is to build the model piece by piece to gain further understanding of transformers, train it on a small data set of text and then try to generate meaningful output.
My secondary goal is to compare and contrast different generation functions.

## Examples

### Shakespeare

*AUFIDIUS:
So warrant, Sir, uncle;
Do here one gentle fet her father proceed
Faith strike and the cale in all her man!
A children, and like me soundly convey'd worship;
But who should pity your royaling gentleman?
He liest me w friends ne're not of you are nor greet of
ot, you fire. In us nothing truth to be done.

MENENIUS:
The besteed, he'll ever she died me better with her.

PERDIVERSO:
To she.

Second Servingman:
So here, but puts you with witding, flower 'tis told:
The fathere shall provoke her beggarle,
But see apolume you to rusing in herself.

JOHN OF GAUNT:
That they shall enrich with King Herence's darth,
Or oft orath! Look and they that thou wantor bends,
Thou so left'st thou hast to loath, pale to mark our:
Whast any mind, booth will he soot it beloved,
To fe my forfe; and I have done,--

LADY ANNE:
So standing what?

ROMEO:
Should it.

KING LEWIS XI:
Hold, hark: shall come the kneft duke off;
He weight make daughter courself and part indeed:
But she would strift
Intercale mercy, if the will cannot shall sit

Put ever: there is that third in the tales of men
Amagia wall hence, on the his body of men,
To prepared in him.

GREGORY:
Speak madly great in this desperate corse;
Indeed from that he list to stay, so live,
Let that yet no excuse, go my mind. Have I
An imper thy Forbalty weath to him honour.

KING HENRY OVI:
Go, be make far, many lord.*

<br>

Some immediate take aways from the results - none of it makes much sense. It has managed to generate new characters given the previous characters, but it does not understand the words in a grammatical or contextual sense. There is potentially much to do here with word/sentence embeddings!

![](image.png)
The model was based on the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762), and Andrej Karparthy's amazing tutorials on youtube: https://www.youtube.com/@AndrejKarpathy
