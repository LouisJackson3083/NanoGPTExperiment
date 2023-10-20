# NanoGPTExperiment
### What is this project about?
This project was my attempt at building a gpt model - similar to Chat GPT.
The primary goal when I started this in the summer of 2023 was to build the model piece by piece to gain an understanding of transformers, train it on a small dataset, and then try to generate meaningful output.
Since picking the project back up again with the start of my NLP studies, I aim to expand on my original code, introducing different tokenisation and generation methods.
### Current goals:
- [x] Implement N-character tokenisation
- [ ] Implement N-gram tokenisation
- [ ] Implement different generation methods

## Setup
To run this model, you need to have a CUDA supporting GPU, otherwise the model will take a long time to run on a CPU.
The model takes in an input text file, the longer the better, and tries to generate text based off of the character sequences found in the text.

Windows:
```
py -m venv venv
venv\Scripts\activate.bat
py -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu117
py -m pip install -r requirements.txt
```

Linux/OSX:
```
python3 -m venv venv
venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu117
python3 -m pip install -r requirements.txt
```

I've supplied 2 example texts to the model, one is the entire works of Shakespeare (1,115,394 characters) and episodes IV to VI of Star Wars (159,478 characters).

## Findings
### 1-Character
The first results I generated was with a 1-character model. At first glance the 1-character model appears to be a normal Shakespeare script, with appropriate new lines and paragraphs, but under close inspection it makes no grammatical sense.
It can predict and generate new characters given the previous characters of a sentence, but it does not understand the words in a contextual sense - because the model does employ word or sentence embeddings. Despite this, it is very good at constructing actual words, just not very good at stringing them together in a way that makes grammatical or semantic sense.

### 2-Character
The second results I generated was with a 2-character model. 
### Dataset comparison
It's interesting to compare the output of both the Shakespeare text and the Star Wars text.
The Shakespeare text is more unique and random than the Star Wars text, but is less coherent.
When I was training on the Star Wars  text, I found that the model got a significantly lower training loss but the validation loss started to climb up as it neared 5000 steps. I think this is because it started to overfit the text data, as the script is too too small. Despite this there are more lines in the generated Star Wars text that make sense, like "*LEIA: Are you all right?  Come on.*". However, I feel this is because these kinds of phrases would appear multiple times throughout the script.
There is potentially much to do here with word/sentence embeddings, which would allow the model to generate more coherent sentences - though I want to start over with a new model to do this.

<br>

## Results
#### 1-Character
<details>
    <summary>1-Character Shakespeare</summary>

    AUFIDIUS:
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
    Go, be make far, many lord.

</details>

<details>
    <summary>1-Character Star Wars</summary>

    LEIA: You like back it!

    LANDO: Backing them? Whey don't -- wrong, we come here?

    LEIA: All right? I'm quite powerfr to compass.

    HAN: Command you too late vacuate.

    HAN: Not really!

    LUKE: Dack!

    HAN: You havo one board scons.

    LEIA: You would use, going?

    THREEPIO: Do you take this true point! The ability to this shat!  Some kid!

    THREEPIO: Look!  I told you to forget it.  Turn to outrange, which will you be.

    HAN: Then they hear st. I lose to your deaction!

    LUKE: I told my gone in in the fire quire.

    LUKE: I'll be just going crazy with you, the Jabba the Hutt.

    LI'm trying compled!

    HAN: Chewie! G--one-three!

    HAN: Get out there!

    LEIA: Are you all right?  Come on.

    LUKE: I'll be at two!

    HAN: Get of her!

    LUKE: Now!  Come on!

    LEIA: I you gotta good with about you.  Oucce make stop!  Where could plensure?

    LUKE: You, but this fightere wars aways!

    LUKE: Look I'm a way another you could. Thear-- you get? Oh, did up! Oh! So go!

    THREEPIO: Sir, that of your shipful! Jabba at madme well palt!

    HAN: Chewie, get up on the security-tworders and not aventher energy season.  We'll move on the leader shield and cannon will give confire those planetration block AA--ythreport ling.  The Rebel cruise well speed from a great pilot of being strange.

    LUKE: It looks like I'm going nowhere.

    HAN: I'm all right, patrol. Now for a droids will if I get bocEdeult. Right speed, signal.

    LEIA: Cut the ship no leave!

    HAN: Ult me to it!

    LUKE: Jabba.  He's that only a fighter place chance.  You have been jettisoned.

    VADER: Did you fire droids?

    LANDO: No, we've going aboard. I just want you to live been patience.

    YODA: Hman change. No disince time.

    HAN: I could about somebody get this big so bucket I could.

    LUKE: Thank you. To younders.

    THREEPIO: I once you \Nice!  Come on!

    HAN: I told you to turn one.

    HAN: Don't to really to picked out by hom interfere.

    LUKE: What about the might helpere?  You know it!

    LUKE: Quietly, see where might back.

    THREEPIO: Jabba offers the but of your sight, sir.  I interructed and in a

</details>

#### 2-Character

<details>
    <summary>2-Character Shakespeare</summary>

    ABETH:
    A bait will leader for most summer law. I'll to discove them.

    First Caius from the want.

    WhicERLady:
    Go, hence.

    CAPULET:
    O dead!

    AUFIDIUS:
    So sir audierate is go better compouts and more.

    SICINIUS:
    Condition against, go all:
    Sinker you'll stay out of Rome.

    CORIOLANUS:
    Nearer sense are trust them!

    First Citizen:
    Ay, series; it goes win
    Here could have beheld him a worthy for his spilt
    In law, no lesser wife will voay.
    And, what you will dead?

    SAMPSON:
    But shepherd! same ut I, sir; and who, office thou commonst!

    JOHN OF GAUNT:
    Sir, that let me stand before her again!

    NORTHUMBERLAND:
    For being drunk them, thy lord;
    Forbot on the thousand duty dug and my dealth.

    GLOUCESTER:
    Even that my traitor woes
    That bleedy patritten scourge the idle bonds
    Of move and bid thyself, and forced thee,
    Setake thy leagues to expediention of such
    And to win the wind to tedious wounds with thy gentlewo art
    the two gorge home: to Content;
    For evident themselves
    Beseen until that ever power-but I say well. well, Is
    doubt, Ill, fetch against thy father, well patient of York:
    If write, if we come then? or your eye-hook
    Shall perist not for Time.

    TYRREL:
    Thus knee not for doubt such war:
    'Tis not much go to excell him about his country.

    KING RICHARD II:
    Even so he that thou pity in thy soldiers!

    DUKE OF AUFIoiew my son:
    Why is the oract that a heart is us stay:
    Which shall my beautiful tongue that refuse
    For our virtueds tunety takes me to be
    some witness; to one badees? but, if we hate our eye.

    LADY CAPULET:
    My liege!

    ROMEO:
    What?

    Messenger:
    Be very pimbly, indeed. You are opposed
    not an untauntimed cred to shore them on other.
    TuRTHUMBERLAND:
    The purse take your grace of me
    To wave at one to bear us us of lawful lett
    That we are weak usurp?

    Second Musician:
    They'll't not weep; but let them be king;
    for, since notwice know him man let:
    I'll make take my gentle angry tilted
    Young landed enemies, your tongue present;
    Hear him to attaste where I speak.

    PAULINA:
    I came to some much that
    As by your first harm cannot do this particules am
    Your band-countaintion this envious shall and
    Forth for having night. Coventry, kill him: stay,
    I'll unget him for God live him am oppoor to
    stnaty to this crown helmed-trees, be marr'd; 'God!
    Rine adiers, corse practise voices, girl,
    My life, sometime to part
    Because their colour.

    EXETER:
    Have not the king of peace?

    Too rashful:
    I hope to supply gift can constable,
    That I vow for himself thy mind:
    And therefore, marry, that alone; cannot yield it!

    ROMEO:
    So royal friends of my tent.
    Heaven, phen ospesy!

    GREMIO:
    Besides my back! madam, and between nosor
    but beavens some days: I'ld try you
    So too grief as they, as my ladies to him?

    CLARENCE:
    Under grief; I pray you, grand Buckingham a
    grief strike beats for his great ley and eye.

    KING HENRY VI:
    Why, to choose the tentastivious weddle angrey
    That you repeal wish the purseman of stoccasion;
    No twes; not from no more messetable.

    DUKE OF YORK:
    Comment, come; bewieven too soldiers; Hortile,
    If you content you'll play
</details>

<details>
    <summary>2-Character Star Wars</summary>
    
    AA, makes up!
    NEEDA: Only movery ybody.  Do you know him?
    CREATURE: Help you. You would be done. For my ally the sattle side of fact!  My hand no sign of th My father religion. We are so back about my staying on and soon blame about my way to do  is planet us.
    LUKE: Like!  if they're been up!  Okay!
    THREEPIO: We're watch your!
    HAN: We're watch your back, going into the city!
    LUKE: Listen, friend, that's what you'll be IA: fortunation  are working if it can't him beginning to take him back. We going to fry iends with the Falcon, kid.
    LUKE: But what that is do you.  No, I want to check it out on along by all myselfunction.  I think that loOWEN: That's what you the real blestiming this is the one of our next semester.  If you'll hu   unable to see the message to your master soon on the deal.  We'll do to stay here much function!
    LUKE: Artoo.  Dagoings This place is the in zone out of the choice.  He might bet us.
    LUKE: All right, I'd like to take him down to maintenance power.  No!
    BIGGS: This is it.  Rouge Three you are clouds coming in.
    LUKE: Hey.
    BIGGS: Come in with nce well.  I cut you off for a few time -one, for you.
    LUKE: Nothing.  I'm all right.
    LUKE: There's all right.  Bring Look your father and I wish I was going..
    LUKE: But I was going to Toshi Station to pick up some power.  No signal... not any  fry of anircentures.  There's not my plic ere are work.
    LUKE: I'm sorry to saved my mindred him.  That is the way of things ... the way of the Force is if you?
    LUKE: Well, I don't want the Emperor'.  Hang on our tight now.
    VADER: Imperial ways been sh have your felt it was until on.
    HAN: That looks pretty good.  You have the magnetic field by the far to the Rebellion.   All right, I've lost my guts wither Dantooine to your abrce  I become back.
    LUKE: My scope can't see a thing time. She's going to be wrong ong the  spice of the spice miness.
    LUKE: We're on our on our only chance.  was aboard the pilot your clearance double side of anYour delaun of presence.
    LANDO: What are you doing here?
    LUKE: I know, I know what she mean e very good pilot you.  You can do about the next year?
    BIGGS: You can subs  on the place the effort.  Come over the hyperdrive.
    THREEPIO: There's something our chance!  Go!
    r in there an hands are no set the corrKAy stupt.  We'll never out of the ship that deactivate has been deily!
    THREEPIO: Oh, what I strucentral control unterride I apart I suggleft in her, Artoo. I be more careful.
    LUKE: Oh, my.
    ! Oh, no!
    THREEPIO: Oh, no.  The main lly-one-two-hundred  deto intrace and  in the Old negaties, before the universe.  I have ferect missed you for a choice?
    GREEDO: Not yet.  If you got some thing to else.
    HAN: Look at me! You want my like I'll pay you back the scanner. Plar back the city cling.
    BIGGS: Luke!  Remembered!
    TRENCH OFFICER: We are seven an emy ships out of the  magnetic field.
    RED LEADER: Hold tight up around for your signal.... to stand  friend.
    LUKE: Dantooine Vader!
    LU
</details>


<br>

# How I developed this model

![](image.png)

The model was based on the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762), and Andrej Karparthy's amazing tutorials on youtube: https://www.youtube.com/@AndrejKarpathy
