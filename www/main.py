from __future__ import print_function
from __future__ import division

import argparse
import torch
from torch.nn import CTCLoss
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import dataloader
from dataloader import data_loader
import dataloader_gridmask
import json
import model
import glob
import evaluation
import albumentations
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
#----------------추가 import ----------------
import sys
sys.path.append('../scatter')
sys.path.append('../')
import www_model
import utils
import time


def SaveDir_maker(base_model_dir = '/tf/notebooks/models'):
    trial_count = 0
    mon = str(time.localtime().tm_mon) if len(str(time.localtime().tm_mon)) ==2 else '0'+str(time.localtime().tm_mon)
    day = str(time.localtime().tm_mday) if len(str(time.localtime().tm_mday)) ==2 else '0'+str(time.localtime().tm_mday)
    directory = f'{base_model_dir}/www_{mon}{day}/{trial_count}'
    empty_flag = True
    while empty_flag :
        if (os.path.exists(os.path.join(base_model_dir, directory))) and (os.path.isfile(os.path.join(base_model_dir, directory, 'best_accuracy.pth'))):
            trial_count+=1
            directory = f'{base_model_dir}/www_{mon}{day}/{trial_count}'
        else: 
            empty_flag=False
    return directory

def make_folder(path) :
    try :
        os.mkdir(path)
    except Exception as e :
        pass

def save_model(base_path , model_name, model, optimizer, scheduler, opt):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(base_path , model_name + '.pth'))
    print('model saved')
    
def load_pretrained_model(model_name, model, optimizer=None, scheduler=None):
    model_dir = '/tf/notebooks/models/'
    state = torch.load(os.path.join(model_dir, model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')



def train(num_epochs, model, device, train_loader, val_loader, images, texts, lengths, converter, optimizer, lr_scheduler, prediction_dir, print_iter, opt) :

#     criterion = CTCLoss()
#     criterion.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0).to(device)
    images = images.to(device)
    model.to(device)
    for epoch in range(num_epochs) :
        count = 0
        model.train()
        for i, datas in enumerate(train_loader) :
            datas, targets = datas
            batch_size = datas.size(0)
            count += batch_size
            dataloader.loadData(images, datas)
            t, l = converter.encode(targets, opt.batch_max_length)
            dataloader.loadData(texts, t)
            dataloader.loadData(lengths, l)
            preds = model(images, t[:, :-1])
#             preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            cost = criterion(preds.view(-1, preds.shape[-1]), t[:, 1:].contiguous().view(-1))
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
            optimizer.step()
            if count % print_iter < train_loader.batch_size :
                print('epoch {} [{}/{}]loss : {}'.format(epoch, count, len(train_loader.dataset), cost))
    
        if (epoch %3==0) &(epoch !=0 ):
            res = validation(model, device, val_loader, images, texts, lengths, converter, prediction_dir, opt)
            save_model(opt.save_dir,  f'{epoch}_{round(float(res),3)}', model, optimizer, lr_scheduler, opt)

#         lr_scheduler.step()

def test(model, device, test_loader, images, texts, lengths, converter, prediction_dir, opt) :
    model.to(device)
    images = images.to(device)
    model.eval()
    pred_json = {}
    pred_list = []
    make_folder(prediction_dir)
    for i, datas in enumerate(test_loader) :
        datas, targets = datas
        batch_size = datas.size(0)
        dataloader.loadData(images, datas)
        t, l = converter.encode(targets, opt.batch_max_length)
        dataloader.loadData(texts, t)
        dataloader.loadData(lengths, l)

        preds = model(images, t[:, :-1 ], is_train=False)
        target = t[:, 1:]

#         preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)

#         preds = preds.transpose(1, 0).contiguous().view(-1)
#         preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#         pred_string = converter.decode(preds.data, preds_size.data, raw=False)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        pred_string = converter.decode(preds, length_for_pred)[0]
        pred_string = pred_string[ : pred_string.find('[s]')]
        
        pred_dict = {'image_path' : test_loader.dataset.get_img_path(i), 'prediction' : pred_string}
        pred_list.append(pred_dict)
    
    pred_json = {'predict' : pred_list}
    with open(os.path.join(prediction_dir, 'predict.json'), 'w') as save_json :
        json.dump(pred_json, save_json, indent=2, ensure_ascii=False)


def validation(model, device, val_loader, images, texts, lengths, converter, prediction_dir, opt) :
    test(model, device, val_loader, images, texts, lengths, converter, prediction_dir, opt)
    print('validation test finish')
    DR_PATH = os.path.join(prediction_dir, 'predict.json')
    GT_PATH = glob.glob(val_loader.dataset.get_root() + '/*.json')[0]
    res = evaluation.evaluation_metrics(DR_PATH, GT_PATH)
    print('validation : ', res)
    return res

DATASET_PATH = os.path.join('../data')

def main() :

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--letter", type=str, default=" ,.()\'\"?!01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읩읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝")
    
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=500)
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--print_iter", type=int, default=1000)
    args.add_argument("--imgH", type=int, default=32)
    args.add_argument("--imgW", type=int, default=200)
    args.add_argument("--batch_size", type=int, default=192)
    args.add_argument("--batch_max_length", type=int, default=45)
    args.add_argument("--output_channel", type=int, default=320)
    args.add_argument("--hidden_size", type=int, default=160)
    args.add_argument("--input_channel", type=int, default=1)
    
    args.add_argument("--lr", type=float, default=1.0)
    args.add_argument("--rho", type=float, default=0.95)
    args.add_argument("--eps", type=float, default=1e-8)
    args.add_argument("--grad_clip", type=float, default=5)
      
    args.add_argument("--PAD", type=bool, default=True)
    args.add_argument("--data_filtering_off", type=bool, default=True)
    args.add_argument("--num_fiducial", type=int, default=20)
    args.add_argument('--extract', type=str, default='RCNN')
    args.add_argument('--grid_mask', type=bool, default=True)
    args.add_argument("--load_model", type=str, default='')
    args.add_argument("--save_dir", type=str, default= '')
    ###########################################
    
    opt = args.parse_args()
    
    if not opt.save_dir:
        opt.save_dir = f'{SaveDir_maker()}'
        
    letter = opt.letter
    lr = opt.lr
    cuda = opt.cuda
    num_epochs = opt.num_epochs
    load_model = opt.load_model
    batch = opt.batch_size
    mode = opt.mode
    prediction_dir = f'/tf/notebooks/models/{opt.load_model.split("/")[0]}/{opt.load_model.split("/")[1]}/prediction'
    print_iter = opt.print_iter
    imgH = opt.imgH
    imgW = opt.imgW
    converter = utils.AttnLabelConverter(opt.letter)
    opt.num_classes = len(converter.character)
    #########################################
    
    device = torch.device('cuda') if cuda else torch.device('cpu')
    new_model  = www_model.STR(opt, device)
    
    if load_model:
        load_pretrained_model(load_model, new_model)
    
    images = torch.FloatTensor(batch, 1, imgH, imgW)
    texts = torch.IntTensor(batch * 1000)
    lengths = torch.IntTensor(batch)
    
    images = Variable(images)
    texts = Variable(texts)
    lengths = Variable(lengths)
    
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in new_model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("------------------------------------------------------------")
    print('prediction directory : ' , prediction_dir)
    
    if mode == 'train' :
        print('data loading')
        if opt.grid_mask :
            transforms_train = Compose([dataloader_gridmask.GridMask(num_grid=(1,1), div_val=[10,2], p=0.3), OneOf([IAAAdditiveGaussianNoise(),
                                                                                                                 GaussNoise()], p=0.3 )])
            train_loader = dataloader_gridmask.data_loader(DATASET_PATH, batch, imgH, imgW, transform = transforms_train, phase='train')
            val_loader = data_loader(DATASET_PATH, 1, imgH, imgW, phase='val')
            
        else:
            train_loader = data_loader(
                DATASET_PATH, batch, imgH, imgW, phase='train')
            val_loader = data_loader(
                DATASET_PATH, 1, imgH, imgW, phase='val')                      
    
        params = [p for p in new_model.parameters() if p.requires_grad]
#         optimizer = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        optimizer = optim.Adadelta(params, lr= opt.lr, rho = opt.rho, eps = opt.eps)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
        
        #opt log
        os.makedirs(f'{opt.save_dir}', exist_ok=True)
        with open(f'{opt.save_dir}/opt.txt', 'a') as opt_file:
            opt_log = '---------------------Options-----------------\n'
            args = vars(opt)
            for k, v in args.items():
                opt_log +=f'{str(k)} : {str(v)}\n'
            opt_log +='---------------------------------------------\n'
            opt_file.write(opt_log)
            
        print('train start')
        train(num_epochs, new_model, device, train_loader, val_loader, images, texts, lengths, converter, optimizer, lr_scheduler, prediction_dir, print_iter, opt)
        
    elif mode == 'test' :
        print('test start')
        test_loader = data_loader(
            DATASET_PATH, 1, imgH, imgW, phase='test')
        test(new_model, device, test_loader, images, texts, lengths, converter, prediction_dir, opt)


​    
if __name__ == '__main__' :
​    main()