# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-compatible inputs and raw output-dtype ULP metrics for the port acceptance gate.

Independent implementation of the GDN prefill input distributions: checkpoint
gate-head weights, projected rows, heavy-tailed and adversarial families, initial
states and history rows. The checkpoint parameter *data* below preserves the full
gate-head distribution.

ULP here means spacing at the FP64 reference in the actual output dtype, with no
kappa, row-RMS scale, rounding of the reference, or absolute-error forgiveness.
Five fixed public seeds drive the inputs.
"""

import base64
import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache

import torch

SEEDS = tuple(4200 + 10000 * i for i in range(5))
REGIMES = ("normal", "heavy", "adversarial", "robust")
# Diagnostics only (famprobe.py): every adversarial span uses this family index.
FORCE_FAMILY = None
# Diagnostics only: False skips the adversarial initial-state zero/x1024 rows.
SCALE_STATE = True
# Diagnostics only: the scaled family touches only "q", "k" or "v" columns.
SCALE_PART = None


@dataclass(frozen=True)
class ULPCase:
    hk: int
    hv: int
    kind: str
    lengths: tuple[int, ...]


# Acceptance row layouts: packed sequence lengths per (hk, hv) geometry.
OWNER_LAYOUTS = (
    (8, 24, (11,)),
    (8, 24, (5, 11, 7, 8)),
    (8, 24, (583, 324, 103, 76, 194)),
    (8, 16, (1, 63, 64, 65, 81)),
    (8, 16, (128,) * 83 + (30,)),
    (
        16,
        32,
        (
            8,
            4,
            1,
            17,
            9,
            49,
            17,
            13,
            8,
            2,
            70,
            3,
            49,
            22,
            6,
            60,
            59,
            14,
            11,
            4,
            4,
            1,
            12,
            41,
            3,
            20,
            150,
        ),
    ),
    (16, 32, (128,) * 28 + (14,)),
)


def acceptance_cases():
    """Every acceptance row in the normal, heavy and robust data regimes."""
    return [
        ULPCase(hk, hv, kind, lengths)
        for hk, hv, lengths in OWNER_LAYOUTS
        for kind in ("normal", "heavy", "robust")
    ]


def exhaustive_cases(geoms, lengths):
    """Acceptance rows plus all frozen layouts/regimes and tiny packed normal batches."""
    cases = [case for case in acceptance_cases() if (case.hk, case.hv) in geoms]
    for hk, hv in geoms:
        for packed in lengths:
            cases.extend(ULPCase(hk, hv, kind, tuple(packed)) for kind in REGIMES)
        cases.extend(
            ULPCase(hk, hv, "normal", packed)
            for packed in ((1, 1), (2, 1, 1, 1, 1, 1, 1), (1,) * 16)
        )
    # Include adversarial controls on the acceptance layouts as well.
    cases.extend(
        ULPCase(hk, hv, "adversarial", packed)
        for hk, hv, packed in OWNER_LAYOUTS
        if (hk, hv) in geoms
    )
    return list(dict.fromkeys(cases))


def ulp_spacing(reference, dtype):
    """Binade spacing, including zero/subnormals; powers of two use the upper binade."""
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("ULP outputs must be BF16 or FP32")
    bits = 8 if dtype == torch.bfloat16 else 24
    magnitude = reference.to(torch.float64).abs()
    _, exponent = torch.frexp(magnitude)
    minimum = 2.0 ** (-125 - bits)
    spacing = torch.exp2((exponent - bits).double()).clamp_min(minimum)
    return torch.where(magnitude == 0, minimum, spacing)


def ulp_metrics(got, reference):
    """Max error and strict >4 count; nonfinite values fail, empty state views pass."""
    if got.shape != reference.shape:
        raise ValueError("output and reference shapes must match")
    # Validate dtype even for an empty view. Slice to bound temporary FP64 storage.
    ulp_spacing(reference.new_empty(0), got.dtype)
    maximum, above, finite = 0.0, 0, True
    flat_got, flat_ref = got.reshape(-1), reference.reshape(-1)
    for start in range(0, got.numel(), 1 << 20):
        actual = flat_got[start : start + (1 << 20)].double()
        exact = flat_ref[start : start + (1 << 20)].to(got.device, torch.float64)
        valid = torch.isfinite(actual) & torch.isfinite(exact)
        error = (actual - exact).abs() / ulp_spacing(exact, got.dtype)
        error = torch.where(valid, error, math.inf)
        maximum = max(maximum, float(error.max()))
        above += int((error > 4).sum())
        finite = finite and bool(valid.all())
    return {"max_ulp": maximum, "gt4": above, "finite": finite, "elements": got.numel()}


def ulp_pass(candidate, fla):
    return (
        candidate["finite"]
        and fla["finite"]
        and candidate["max_ulp"] <= fla["max_ulp"]
        and candidate["gt4"] <= fla["gt4"]
    )


# Gate-head parameters: 4,032 (A_log, dt_bias) pairs in checkpoint order
# (9B, 27B, 35B-A3B), little-endian FP32, zlib + base85.
_EXTREME_INDICES = (1229, 776, 3094, 3101, 3083, 3090, 3091, 3278)
_LAYER0_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
)
_HEAD_DATA = (
    "c-l>!dAyAE7soGd_MH;h5=m0ptaqMgu0lz)38|1$M4K(Ctl1??*_Z5Vq@uX<%%D<sC8;D!S=zMG=9l03IPZD!$GpCG=05W*-"
    "|tyI=X1{YOh(3)sf>(=>{Z2Hw`6BzOtSI4WJbnt8ynfHp1mg8=lyJ4Wv{#JH9ww_(Za^HcH9sf%h+q2y}q^ggKf;R*9d#nvR"
    "65K^|9|?Y-2YY2iiEpjvK`HC)k*&YuM`|d);HNDJehyEPLg$*M0WAzwOw-#)s{_?)gEEZJ_UE`+Z($;}bUCu4~!rQhPnceLQ"
    "Nf7wuI<&tS(D*7MoeD9p(CGRVm2VdKj>uRYuKHtw;{_NDy3*Vt=REF+_fjVH3~IJ=kM?f3=ueoT~+QQ7|9Zl4uepOKN<zQ4="
    "<&bG0<{r_O-*Lui4+ic^%HZHU=W#e+qxs5Aq>~6<r+c?sWUuB;)u=l;~{doKPO?!RFoay`T*jUrXZFZ04?6^zp{ag0Eoi+}z"
    "@iqJ3A{)Q3v6TIPlzslWz29Q*zqWCkjk|38)5hEE|3mHnt!#YB#)O^oeH%;Yc+J0kexc55<5(Lnu<^!}Kj(J-uXQSG$A4z$d"
    ")?+~uDvgApV!xO+c?t3Nj6Xav;Sw=aZBy}HTL(dT0{H)OdG4&SVwXg`u*;=|G#SE74~@>8?TT1TISgQi)&5n|I=)I#y)?=#>"
    "Gj$PWpLo30<ZZ+c?PnmrVL~-q7_W-*I2>3HE-qy{{2>8Cq)ZKhg0vK4<U0vGEpUHOI!L_8fmFGBWZqZ`)Zv$*|6Ufz97H_Sr"
    ";x&9L{6*|^chFYNQh_8MyE`@r5mZu2x$&tUI&>-lYLp7QmSyw<b#C2hRQ-rs5Cwf4Th9UqUoZcgAi3flX2lIhUp>Xdyp+s1t"
    "Mnk?P3&tJ9A_S$$L>CdM1N+exBY6bpWNxt_;<p0ZMuRD@1>+9_E+V=e;Hok6SRU5b1_Zr*(KCt8S*?in=?`PTHx7+_ew*MEk"
    "v5eN-p0k?$e~OLACCm2t&C-j6&)Y^j<|=#LY5)7l#sK}Rk3M`Y8I}G@?vt)JuS@Urw~g)Wd`-9?-Dd@RpG>(9YJIY7Y?t)+l"
    "BxFg+G)q+kGp-Vo^<)X!LHpt<?AXNw#<%sTeiqPf5wjK&v=ntr=r%|{{G8e_cwIAQaS0z>%OG-!|nS&+ZYSoW<A0BwoAB8yD"
    "W4&rumgT?+9J5^CsK|&$R1YEq%A~OB<iJ`>L68JD_zPVxRAoo+aH5-EQMo_Nr^gy`krk{@YmE&R1FX%kKXl`~Ui+%U@~x|2+"
    "G<sASY$AE8f@U)hf{lDCx0*8m%rCw%UX$+pNI*!gnnI`%$)=<=43FG#waU1qP{NtfG?6MmnP=c=+>Hm<}DejoaEKMUO+%C0Q"
    "7ab42=%@2|p8$YmdgN@md%kvVm!|x`2PPEQr(a+iPLH5dHf4?9*V&{F_#=;?X)W-Gp`a0(8bFJ=Q{>#qWP|s=a3)t9O@@TKm"
    "u~WLH{7nh&yIth-JTdh3NJf6$I!U)<9g=P*=W3prC%w<I`Ta@uKXiXk%RWD8ue+0Oms`i(-aMUfU-qSaw%W!sHm*#$4(yQq;"
    "@PFQb67vwpA$*9lYcUQAKP{6Y8~x;UmK?}pC8-v@6tNheSajsll14&c{l4`u%EI6+33T18-GmtylP$UmCv>N{nW+>l0MG|Lf"
    "6CA@->oM8|&M%Ty4+&iS*NspN<{gBRhyclTK>>U$*<%iX4^Iddf~jZrAmkPup=n>T~>4BIfV&*zuRkU)Zsy>^mLo^C|XOlw("
    "-HKHH98k$lJxjSbzuEVI9#v2jV#uY0q7?|bG+cI<xU_i~-fUPa<QkKctp&-<}6-z6Ob)s5WVcMBaG$?nKLXl-@<E=iw<F|5n"
    "Rq}x*+_pJ2P&fisbKjm^TIpIEJyvcjpl*`yQ{Gaq;edN0GY~YwlerKG${;<ynMQ)>>PdR?leaw~px7UBjf&7>JeK~vXoTS?&"
    "`4{;n{jUg^ZX@53|9{Qi4~*O|-5j~SxHWY79*({oL0&&Vt~%IpHwmZM`@{OouJ=F9Rp|DoGFVc2aS%JYS8IZOmw$dwSkJ~WH"
    "ui}5a~?`M7Q9QkZDZrW?d?bSv#NIdFdN&4j!FC3|EtTdXZgCyhRA0w5~fS~{G`Blvgd{4j%T{rSPeWUegDeF2U6~jb>0s`mx"
    "G6GEMd=8EalhHdI@uW5xM`<`TC^XW@`O%n2#SDm<$Scf{D-M_;n=1@>BZ~u1~@_75JXy_u-_UZx8mnEB7aS@EN#dlf7;XT%P"
    "8D?Y~dDj7iR=6S=_1TJuuiovq*@op(~wb-q&Q>-wv(2>4uf|B95~_ip^M?51$@JCU!0utD0Mu0)<BR|CP(uV5c1VrPYWH$-m"
    "hbp9ow%e^qqcw74sNw?9Xu}AVh^1GYBTU*iJj`BzL?4R2AL;Q;LOW0UAL2EKU<u-UK@+Ew+CG_Wi6n)snyb3GIK3tIU`Om|="
    "Nnds*{rXv~kMMhO`(6k9=u^zg_>^O1&DRF>?{w(;q`2Wtu2Y72(03+*6YfRMG_Q&Wn#c~@tGnWl$Zg#oo<sgr7_FlGeA0c$R"
    "rZ=9tYLG~EakC@<ohrC-tRWHMIOrHhou)f?ib`{tI1*c(4SlDb};h!lbunFC?BhNKFsrH5{DJY@#A#vFR;Jz7rB#eKN|5~c<"
    "3hdS2%Te!tG@;=Kf9crstB+mVUmNbU!6IxnFiOblfeh^Ihcj=-H%Wx3jhhn<2e_De~v~EOJaIJvy!9BVT{Tkk_RAzV%FF(dk"
    "*lp!4w`W588!+p9$C91dR<IcC2w>DO5fu9rS+vgf%>=g@P8=k7PHKf-cv;m4P<F4BMbuD+3DDPd#1`i72SHE*R;ZYPE7)@qJ"
    "o?r(d6TZV&yUb3+#v6N!y`{egS_qneyzrroQnw(6Cd)&DMJ)6Ti=)8qeeh%rL@S-rG^k=#q-z{)C=yzUhFI>13EZ+h7n8)>|"
    "<DW(Z{(e4qrefrFFEj4ntwC(vGUfArQRv)*{L)f8wtDD(W)1UqPv~<i%q$=NO6dBjXUk+xb>CMc{k_hg2P`Umo@uY6=#ZXYz"
    "E$?_0q|p)xX*VJV)Z?t+qYJs%l>sKU$1AF|93)vKKYG;JjX@w6!Nh{l5VGj-Q-8)PbFKL1IcMDa33%$a9>p{^mVVxdH&6Dy_"
    "OA^ymrODDNdUdx?W3P6~jm_8ruC&#WyKdxHjeEO^kQ3zU|?9^gILX_}X@S2G61E<-i^3e7f(WVC(jQ%Sk0-jy7@EnM-1BN40"
    ")=C0n86%e4N;PU<|eJMu~5jugK>67zW#-Vx?gJXrz#l^$IN?=%ScyeDwI9t#)K7y0~<^~pr9dnDa&lumlQCVoKkEnFacEMHg"
    "!3|^iXW)tz)?UBb-vgdc%^<J>?J-8FuQq9}6z;V%C$oozBQN@b;WIsY*C&`cW>v8;Kv(W8<IGYlDFNfz*tWpiVoErK28TNT4"
    "%_}iiL9p~L;b`oO;w9OA#d9U&Za;qwJyu?vwqszxLyF;Sye{tRq?kvz_!I2TZ2Uo?xZ8rqVjeH$$#(hKiyV~VC7+=+8Wy;`4"
    "`ZDqr{YeIasSs~<Ak~O|7SwSIWt3Wd))Q#RLn7SZsawCdF`I^SY`tA-GTqL3>`n@i+lW}c`U@d3J-{jQ4IGizHle=UqW(i;}"
    "$r#Hc8hzt*<!ED)@z;!BCp>nZ&h{Pw{WHh1mjMvx$+9in;Zj@^+rGtcTY3IpP56g>>@e(BlVTaN+p@$bb9Dxq>eC|CsPFoR-"
    "d0kr?I@u%7JADD+?k98)Q5_=eE$XF=rh{cXZA+u_i8&>K?jH^fy43;xUTvR5~P(=y=%wGQtlTn{B91Cx%+FTk!Ut{oZoYipe"
    "^BL>j(To<_?kUZ+X#Y1VGCrAEVe}q0?v$^*QN#{((fogsKfWuSFD7}bdi{wAHB2OidHR<)camU4l@N2T^-@=VNAw5oboHr6~"
    "dq&{vB92pdWhnDG4_)le^C)ilQ}Jxv?aWQs|3l1KEbjhjE%t5+_P<f&_qz&v-UH0?1oBmp>*yX7Gv5s66wcH7EQ<TQ--8}B4"
    "c(7^61g3cU8pCyk2_cXAm<e?ehudqwjBY7*$~Y2B^Z4|w(IN)`0=Ua2j;}xM;wL&*&O%p7l0f6lsT*hmb!)O$o39HmUSP!vt"
    "5_EXS?1Gw()Mb@(K90am2u@;3Sq~1E*mJV(|8Dxwh_WblMJN`!UUJ9Hn{9_87D@Yo~d84qS9I`31>P*=)bps@aY;H5Y@yw~F"
    "1{;Ty}rNp<7->Yx{`vt3@BW&5@6wsEqpO+B+;cZZI<UJ9M7dN}TD@xQ=vUvqok6gwh4n`@u9Chm}*%u2Y;IF5gh+t}6KKWS?"
    "|CvXf?Y`ysu`}?8Lb$Sncp=_sY;U8d`f8j|#hnJJQO#oj-%(>)Ue9(!Q>)}J>ggS{QksOCUw~CLSz;=tfd^vG0r!fi~p*5Kb"
    "Payl+Kk$1|{3V}L623${Xfftxbm+0&bacJ9jj!AIGrZ1se6GB~vpk=$_GjchG=2vb5l(1L{^Mry2P27d^AZQnOn5w97@yuV?"
    "$`MP8(0Q@;IGj6u<ywmd`Z5e4L;-&_>&>vi!$*2zuP*!D;o}<Fg0+Sd<WMNW)v^{X3XQnp)sF-@#M==eqQ0kt{i`H;Bm?!;w"
    "a6R@KeRO%X%&HOTxO+k$L#srP!go<gxmYCsBN<?>`#%`=}ImzUDISOFl_DDVr@>`2e0+@+^${g^h10=I6MX<cl=V2f_5p7i|"
    "T*&x2D|{zZOHe7$gl?*B64uFdd<t>o9mF_I4vUK}reFyU*R#IDZ=yXpCQg&sT1hTl`nQ4GFPyl7zmFBrK`ku6Ysx<fWEa38c"
    "Lbh~|rVh`E*z~@o?&?fRW;%6TV-51JN1gw|VFAV+KdLH4tdd%|z=IODx#|z4b$oE$a-QJ3qe-yrO9`VD1)HyzH2=grrDLq!c"
    "Y&kLa+wf7cYyFbGZZ~k=_c&kP$a&S4IgaV_Bpj;`51fmN5m$-VIIj6fI)D5*v1D!=uTMI^D_j1feeZ)D$1sQN+#~ILmn2<3m"
    "+@SRLrS1G>jIAv^MP#@uWU%TkClDBKl11JD0Ev~HuB>?vSThtIzQ4q^7(JZe9La$&OFuvOEdyAEQ0^7#(ZSqkCz}PD?*pIf~"
    ">dZw>9#+5}zo$JuYy$c{B1kngH)t8*bqHERU~q;d2j1o(mm;p1)*cNBH=fU`qLqO`)&r|3bG<_vUym`FO&)bLAn#gH8$jxy0"
    ")$22#GR2>CGK-ooHB#Ry~IX=3C_Pl076OHYMv4@aO^0~mFG(o^NSYiJ%4e&55A?(?LF;vMEhj@e2@&NnwDUpPB(UZ5;_sTZ("
    "&pJllmeNWu?Tf+I@?bxmQ<UbTM2tSM87f)9??(1<~=z02PahKme0{0h#0{8L7;r*7w3qD02NO))`SfLmEsjgoY-(4$oPBJ6z"
    "c2D!G@8rc#%(wUVB%NP=ggk)O>5{;oU-91*{P){|^G&+nckKBND<>DZ-^)oj4$ThSriuS<q<beuQXDE=r+L^5fB7~s-Oe-yR"
    "Xzs4FTeIWc77WCnDq5q<Y+(nT;YNB@G;YnUt!_VfzR_L<gqV@*OkrN1pa-M8V=oy;z8k$qjvoFk=uF2;TMN~j4*dA<XF5*-I"
    "()?k^MbbzB=$2Pq;+u^k2fU-=UcEG8d6&j3WR1zR0oDRPw@_*T%@V^mSY0^Q<^Y{#5l2!hOn9OBWUM2oK&Eb6l1cIge9+o%!"
    "^X;|D$OZIR>andH@_M~Bcu&EHpc{GXD`h8`DHi-?)wuI`Szomgtm^R)0}-1R0w{Cxm^ZwA=4I^51Hk?ZB~$k*yijyuM2AL3V"
    "qk4B?6?}E=d5Q9tpGl(DOgG1uP`~Atot;8N&hP|K6d5R!sQ*%7#|AhGvZ!G_KCUU;&ZSsMVmnzu7;rQ2kiJdzM)5je@or;|Q"
    "{zcdmJhGFzn8V-=$>#t&->{hHmk&@sA%Br4?l`Is`JwvMeN2epp~*irLoa6W{UotkcFg&i+q8a>=OHBrMFXE3#lBx*?>~ei_"
    "(gFF`9R&jbYVu8=ZzJMD=w@Nx()m`aQu<S{$D2W`|VF&Req^K<o;B6=QZ-1*TBO_f1d<z%SWotNcqxl$^U5n`oYf+1t*V>d<"
    "|89G>N?GL*S(+!MkZbb{71~9P$q@6Ep6H3lKhjCh$D$UF3<^6IZ;;@m;}1lBY)G1@@~h2R|?pd^iGsFcDn2mN<H5-1Ssk%RP"
    "3_-^Sc7ln6YA?!mfE<-Ud>7o`Hnl9j2S=n{DTQg%%oKz`-J?0hxwxxzeS;Q>d1ljP4H0Rz2(+!v)rXDNB2Gg<C)#63tZ#Ho("
    "MALl_{g?EodJ|CTv9_Mw9xt}XbUUFdIasN-)nK5vWZO}v2Z0SCyfkVE-Z^+lQhhJHWpS~H+M){#dIgZhG+4sv!R=6+4aV^NB"
    "YrV#Wem{yUkIMc<uA9rjso$a}@>{ZFs>it>JTH4PGVnDk2+m(ap6*5DP;1qOdWvT7ld>yk62AV5k7O?zh92wJMh;(0IPR3b%"
    "8u0x9kcd<qb@+rMmg#(R+BSH=V6t<$>IG<>gB2<_cO@Xy$%lPKwS13^?b6wU$B1li0>4atV_7fmEZX)aO_tc{`f8CSAJKvHa"
    "9rn#?a;GBJkO8xS1|l?k84IPqZF?l*;n^7*3u_>)TmeLypI@ncy(#gt)<R#MvvzyN!ta{&V2F`X|rv1mn?P<@*YS=kh3uQ*V"
    "g5zR14_mtBWHcpLqkhhEEuNLPMDf5oNA7vu&HPD1Zj6E}?FI$g+{e+q9QJzWpqdY^cl(Cw6bnEZ<{(G0F9JF_+N_q}<pr@(n"
    "t$q)BTI9C`1uFn&}XY;x5PdRTT<fAe9=(*(A_P~$odOZW@8JdzO{1pGC_^}WACz)40!VBo8VxFThkISE+{^n}(v7^z0EzHlC"
    ")CYZyzDc%|Y20b=J4K#j6z+c=|1Z1r5`KIo^%--CYv)qO_&W8D+mUA-S01?*eoEKT41|~Y6+UzZbzz^7pB{w%{6oCkhPr|4s"
    "Y4w`ordD8rGew&lhl8;!|skl?$UJ?KVzrbvu^)lM_vIxG@%adZE_D+#yvmO={&xtBK08h4|TB@`y$uVQQ*iosayM(y03YW%h"
    ")LVyX@`~=4B&#+aT$+FO|TBXLF5_+eYDmmf*g9fyYR)M{h(vfAhdGsw<F8lm>UG-d4I7CY=x63zxZ+`K+UOH}HFvJShj+ggi"
    "qI`M(L&QwgV!LLPIGZ#_aBDtxCp)Dj6_Z~5<5%xgaQ0ojAn)W<YUx<637r|+MRe2v8M)<<6F<ClfU4xpze5-wjKMUH!pAeYC"
    "v-><-flc_7y=fWlT!o7|`?}ZVC;bcD-z!S-?E1s52H&=aP($_@xTD7?H#qZg5f5YZaCXf9b^&`TX(*3=W$N!Swyu?&B$gkhQ"
    "d<s)6CO4tysn7cue2?_=Pkg=N3E7#I$#e0V^tc=~Z@PcQbZf}@DsI}5<+l4o;O84e+|wTW+e&^ha5?Tvt;OS^pEn1)y$(KM6"
    "F6D&G(YD0xr+JK^_wssxzO8b)ZEl%uI^+VmjsT#Hle>d{~N@Y%_I0m{7;j><51PF{y|LEAoO*bK)yu&@}%lM?fv`U(&E(NAE"
    "CavG<724oqNaKU+D9lf!9+1NL}t8cqGLo%aGp^)Kv^27toY(5q7CCHIOs7?u+D`yOaMFA1eDPS=Os9^>`b=3TZu=L%o3Rv#y"
    "Odhn|aA0B+l1KKDWSk(k#RiX+tXU8=c@oWqo?bfo@Fwa&uZ>tlZ2Dd2JCP(KR1Hs~$pUvYG=!0mcV#T53Pl9AWxJQ@00J{oi"
    "WO&p2v{#DcgTu*#AiFofT^0pn|)M~?p3e%0HUQBY+lUhI7s)f|H2<LYu&-NkL5@#sfAE>q>;kqmQT97<yBjjQ+`Q-<g&ljj8"
    "(fzFr9CJu6wT>U#d5a)F?a6txBrnj1I781NyLCU;>45H4#V#oZPRpO<rfkB`KQ7_t(H?<(N#|DcAUE;>ijBqRs1|q%^}40d^"
    "^3Vb<@r`eo<HkOUEFHwOJoDy11FS0-qLmP#W;R&<oxt^;Kq3|Uo+u8{r}>m#~iuggT6tJi($Joe{aAiUV)r#M5n8YQwCoR0z"
    "YQ5M?t**fynWd?p^tufia)EJ>VYcRY~eV^TMaA?sf_F&S&ut_2e_r(<0Q*zKva9LY}4<cuI5Hj@l=26mt_U+lo;}f+IJ8mB&"
    "*vvz3~(VfY2r-X!45^H6`Q`+11_5w~?~!u{-G>ad$pdvhaojuY6IFpjY$b!*wwDE>;lr{=09cCSA5$Fc*8i{8OLEJ>ZK^OZl"
    "8E{oTjOFgFe$LpzAQ{A@Ws6p`35AwP45NGGj?)x9%N$L>AQ;Qe46`r>}@+><dTh<Joe`e%3st5U@)A)r8s9)O&pC=CCEn?W3"
    "<TQUE#?}4&hF#r_Kd->N>pZK;^C&McHFTeNCHS|0cn-&FeySu~j&_k3$pqiehG$Y8{AT>%er&=F^h35ve!C95*amQWTkud%@"
    "(>;28zeWXY1qO12tS;%?|sAmh@I3Xd<XCLE_F%WwAPX1YOO=<gy)r3P*<%uwo2UXaWUp&19<yE=2tax+o-!Q34f!wsJ4x#6V"
    "8uL$4^wC2IMIG{x0?*iQoG@;k=8^tNN-1Cf7eGPZW6$;qENAX^)ybbhW>i+t@JjJk}WWTiCTOa{fH|vaQ_L5%A^))<N-*@^c"
    "SH&Ofa}&nL%x|IIY?`8M?LU-+M*_B=gO&esXQ{{wD10Wa`7^}?Tm_b(3|-#kfvtx(d}>2LnOAH08%xNs<Oj`k~M!B=ht|6Yv"
    "$>IZI-9={a#x^u<S(t|PN9jiuOXIBBdTLC=v{CRcloA3{N*>9mZ^LUoWm4yShT_@Soqq@a83HJ*_nd2en#XTpedoKci@E17g"
    "5cqUedjCP-zWICh$2`LPsZLCDc5mQ#GaFt>@?MxYR(>)G|6UdTQ&>Xre&vMwHsvW~rwSz8w(VlSL?P@zFL3{2;)4&Vm){Hy-"
    "O9ZyuW}(Zzp6cxPfqhQFH?tpKl<<uaf5VM`@??49(E#M9ATH^@P)eH<B?;D^6(^Q(Cbd*TV<mZ$Ic5JgKr^!+A;F=uR&h8Ao"
    "#ct=e>>RkzdvGJr_7nG6oE%`kT}65EmhLN2xt}k9ko%b_)CaK-~4Q3_Px$V+#9J6t6GDZdZlJT1|Y}h<$s?J*<hj><aH*KrL"
    "ir{BlF?XEOHqSMqPm0+&x+w=;D+C3&tlz=ucq-ca(Xlc@iYy%7de9=J2{t7@MgA|KEIK27nx;-+fwMFa4A!YBW+Ppu$4%rx?"
    "*BjD*3Q;RS7J;!m}LF8vL=gCWr%0cYHc4R)2nwL&swoGEEW$5{2=4DNIj-Se;eqa7n{P$}34DpuNvra#uU&0ZS@K+m=>-VvH"
    "`u_s_U{Cm-#gXf0ne+T!g1B@9anAzsmn)dZ81pfky1~BiKjO;!!KY0n?i<a#9b(^zaQF1cbAz%Y!k0^8{(dBST8q5#6YvD<;"
    "iGo2?mNMI8|)c&<Bx>b^F(gnZf5_*2K@f><b$SDGc^@CSb*O=8y8Z4B0hE@JcIDwM)qY)hyRhi`wjd$iS;?ib((?qzD8F2;t"
    "%cvR}3IODqdq_;P^HVbu+r&UC7Hcc);=G0|vkceFR_BhV$g0XDh+=!g1G-ubqs%EvF_i0@oFWPuRo0hG%&$t=ofeINwAb_fB"
    "Qs=^^w`a+a~q?UQ1LJaBpW;8~PQP_3zIkWzui5wdX=sDXN!{Qoc1Q!FB1plcUkABE&i&$@?w2D;}lthd(nFnfD*!9VCZy9bU"
    "-S5SW~`#3!TZ->18#^(p&XENBoej0m{n>sXc5aL?qkl)DxcQ%5rP`yNN@_gbUy5dh}5SK{q#3Mb9{W_9x`#6>TA`x-yF8t#z"
    "{Ln~v>l5ITdB~~al*RA@s>8Y|^v@T=lj*w4h)Wa1T*5l?r*Y!=Qq+b%hcE8MzMPWCX^fiaUd(rk$bIO1YT{p}c2N3$HN0m%{"
    "-5T9J|GTF;B$w=1^vPE$v&y3<O2AEnsC1#ftR;ZGyMtt*5&N&x)XoXi}?099M??h`%}#0I_%pvaK(P=#ZRttyr<_{hP}#(+#"
    "j7Lx73RJz84JN04`;H;QnAeac>TBtnB=Cf#bZk;NtHSZa=562V8Q|jaoc$iIT6{>?K%){N8(BE#(iKC(Xl%Um8O#mgeIs``%"
    "M=&tokF&(6S}%_L4Kj2sWZ57cBo(KPai%6T8;e8s4#EJIytLHLAw@&C)gS21c`bD^&%@sB^iL+RL8;U`w`z7_j2E{b_RMD_n"
    "=;C}VFICb$I9novq7U^8|ERSm@Q#T`SS9L6HL%09y$UA>Z4OwRB`vEq@-2bSCMjTDK(D{eg13%9KY7qYfLwCo=mIUt|rEXWp"
    "31_H=Lh{-cnZ6hwypEjH-HdJ1HR#f9le3HC&V?%{zca^kWwIYXPzx#S5K$W;n{pfL&<#HBtez%a=C>2?&5wLf-ucJZ$9_G`G"
    "mIf7-b8)$m668@74f$#;j3CA7xI&;8SBJ67C`Rzkz=k4e^G;))YjA_e93rQjz52Yc&iuCRo#zjPttPoL*#kvDactj{N!Zbt8"
    "Qaq<XG@C@(>qbH;PctDt=eFoqX(%&j8<z1^@j^y@ups2t40L;_T~@gIBQK%dv-A+Y_PN=yBB6oZV{|xj)&0KV6Jn`xM@1CiS"
    "k;ANi~C_{CY+71iq=q5kJt?D=N+{=L{K#i@Ed%ySNbld6xMU5H<=M0_QFp(*n_kn2o@SJQmY26L?qeSO7s3bSb6@Q>u36{|j"
    "vjjK(qQ74{zF)}Xyy^k2ND1NdfGJh+)+x5)z3i54l!jH)w-AbNJYpIxdA^7Pj_VBc1AM71G>wMydvE&+Z@N2V($2Zfzqix`P"
    "foc(sljB%UotA7yr?~qd@fh;gP2(Of9R|lrKi*;;-XiYaNvv5J{h!VpEXJ<M&+Ws$?PfoP?CP7?*g@2rmZR=aaw$&#R_Z?GX"
    "J)eRKOOfjWZ%-lbPYXm_&}a#1N_-q_C#C>C#dyP{MZ2e^X+;4X_{guRTo!+{s48sABqo!Ki0z^>HFUhFRuib-fDV0KJYk3oU"
    "3X|bCc&ALjBKWao5)y()m*Ma*ZY5eN)1()1REmcK9Fhe5$dLe#~aCQkyL2%??tZBRsDh$=N!!n6LBy*z<QY{H5$c{)ETr;?f"
    "TZ{{@cu#0_p^pRRbf%VVB*7Up;XoGV?~Lf?lD*q?30*U~T57X1K@m)=*5Iq%gUUb7DJaSi#nOzOFhQ6H?jz)sYc3_q_w!4~%"
    "AzQwu^A#bR9##Qk4Pk>WKU@r%9zaLW%dL8TgC32LXxTQPs*;Vj8UDz8Te=#iXJi-^$>9$X}Jp35Bt(i>S&wAp*{ycYi_TwvO"
    "lTDw89_&x(LQdhk9QSuK&WmGnv)}S><S+v+rv!bhexX;#E6Dv5fpfu<mG|k}@&fU1M1A5&@@fOAbJHBDF7<KZAK8gZ=$Db7d"
    "Z1#|9PNrZ&mexfJ)gA(-@J+at-<^+rS9l@>|45STNv(3?eR{yJnaT+-U}}^2>$me`t%&Z-w1oE|HaqvB8ulOrT%9ReK!h{N5"
    "~7Vx|llp;nY<$g_qk5kGm@2<JaIkVQ4*fP3mN{XK@()K~yiHdaokvAD8{QG~u{&EcMX~;4_c0hKtx2eiFZR4E$G?{S!~K2Uz"
    ")!`S4!T8QbFT#<8D%D0O|q=yUZX^${0G?rVf^e%4;Zz<tqv>VUogqsYf);OFa5|5J*3JIR#lvIYlkhxXD(Vmoot1MA#=j-zh"
    "mYj~aZ@ISKax4`!lgO5n_w91n`o%H+N3to<4N3X;0-_O37LD-eO<aJsEj?46XJK<NJfQPDhUXM5F?-k_G+Q9QKAgA~hIrHu4"
    "Q%UN0#}eo70-Nf-9)OR^MVyj{b^Dfj#VgqJb&9;hr{u$SP(QK_yFCCub#1yPAm-;ki61;c-ONAau^#98uTWpmihA89@b$f@<"
    "5>XTk<Md@5Bv!I`WU-4AO^0WKR^NYXgo|lx-|Dafc^~|sDIeZx_(H0UbSz^EniQal={o9A-<mu?vlPY;dy4WE>oDVMda&*=i"
    "Z0^DNnva_2eVi|L_|9Df)80qukdd@X-nKoh^yu-^0E=4e!<robe?0Df?6<M*fCAG%Jy#1Icqe{zk!D^<rP8=5H^3TC!Qkrtl"
    "^|6Vud*dwe<$9)3UdND=l)`*C`ZSMAGwo1xT!R-zBt)P&<#)dbI_Zlxl9Dyjv(4!7dp??iTAqpm`A<KhzP1<pecBj=NsK544"
    "!`4gURIs3VV<toR|^?4XeU1B}<J&we`$d3Oe`^SE0t()>PiYG(-bwT2i?d)H8pLNNPpIS_P*@eW+>Tfdx{gv*uBiG*$et-LU"
    "{mgn&|I?iw2GY%W<TG=_k9^Ge)gwfB?8TVNTW|XP72vsxl20rOJ{=AY_=5YLfPKnGeoOM8eYstObNw5x;T{Hp?KN*LlFnI3?"
    ")D^n|LGfGBJJHN!~DoL{T=x{)Qp_Z?~MP}KK{z^JI#no()B)9v(NJ?=D#}pkMhKwi6K^iuNP30Hjnvv3$Ab}etmRwE`R?Ee3"
    "bh2oXs;P9Mjy6zxkIvh?(p?c!xe3C&4jCz}acse}Xur74_4_cpk~a{q$<vO<rAnuL{%KuL^m!&*1<okT=r**KwY+{nsM*L$6"
    ">jJEFHQkvlod-?1m`efInbziWMd<p1T)+XH+dbw5k!PgOkT`-@A$yVQ?)e~^;MSxI=fbRU$e$nzEOKmQRgmEgJ8!<$s(dmn;"
    "-FQdk*5cMIq6Dxm%J${M4F6FsS8NOcyIS?nHUQcD=^@OjVqrXY-$n`;cNoCh^2Od{fqn|)??4oKIf1tKMH@y1i^i5H{&!wT;"
    "ed&uhofnu_)fui4rxmzd)ug|}d+-3|;H7%fx8r?c+pE9^#Z%{cKNbPY>wAsk&KEy|9WQ_lU5)>KG~qdlj`TsO0vD_2DoajFI"
    "I?Zr=lu+OFo(k{ea2qakbUR9IZqMz;q)HSG2ph}h&8o0y9Tc-*cW{vT=7x*CtOI)Vny~>OAcN~)~=?0s0`=N34Hyv*Gs(hu+"
    "Z1}M(ky2`h>j9UavRkiPbamu`_nFDDxrz_7Hs-ltWbxO}1qkd{T4zKqyzJTE_d3gB!^CG(;ZDaIf{?Fx#<Ty)E_4b;-jj*HH"
    "l*dLg){GWvOvyx?c_Nx6uaSAI%+hK|GKmj!Fe&*W#1g!bY}2lB%owgdy8M$WH<CwvSZcPW@<95OW%95WjG_#)5u9JuB;dQRS"
    "s-P2x(>hyon-pQWqWxs~?s>L%8Ku-%ID;K~EE@It&pkGT@{M_f*%fZ;|BB^s&sV2Tk_rE3by5N_|Z&ibP$PMq3pl`=Au=0z1"
    "e*xoWFyvHvM|^-ip1{~E=Jki`<L(>Y4_pr}hcA5%Ih36cj(GqX2wC^?@Z{>Z@i+EV_xT?8wGFxNO8)3C_||#Y0p*mBQ5*0#y"
    "+mYRCZY$;IsY}tz4r3wLU&5D2cRY~+XZ0rqU<@-y3L@kl`u#a{%02X`nurITyTcEkH6{bA<Q$Ao?>N~-wOEl|L`OG=u!MRcH"
    "k8Kiq7(w*tMszCllynR04VZg1y04#hh=<o$c%KP2e%%V6Iyfn{othP&GW73*};D&zsY~X(amh2!5+K^0JS;r<dbXg{kE~u3`"
    "_ra@W7JM{zG)m-K63+@EI%eOLOiH{&m2jLNL@8}Qd%um=T+BPQVg?jvp;gG`U29=sU+qAm$t@18_YveCoU<dYQJWRn-D!gcl"
    "B>q5u2>J=;*&yU^OPcH{?q1xZ|DScQJs~#lZDtV|L_vcBWHy2^+^02RMCHlUZoWK%nuWAZ3|7{~bK8qff_vg61sK(xW+3^m{"
    "cLw`w#p~X{9)EqWDg9@pOS#~~zXR92K)y<Ql4OtCXL-$pVuY)yLF+>wpaWc2_FFk)*(BlNT5wiZPzzWF4xkD(B9+mf%FIax_"
    "7Gi8?VY|?G;;Ynnd5dUK@Ugm38=yx{g3sMET{%66MdS&I$e!D8U%k{8*Vxij$YT<O`jaqN}fi?=CL2F61MtRuDhGQEoF!iFT"
    "uv^dGjaF_3x?;CfULsX5Ig$bl)Xnz2WR>oC$8#{S*erDgU_zJEZ)>SM2qv1h<@*y@rZGD$tLpQ2g9{{zU(w;^cRB(CeTa9QG"
    "mZOLG&&9Rn4H*G$CSo+;m76gyCx{x8aVG)wQLB(FVyJ)WwUzJ{1E4p!B(u0pQnQD4&+IqA)Lo8o^SU|-Au_UdNytlQ!qqwc1"
    "T@us-ryhG$8-sK*b;@=a*Q%C5>^)_`c6Jx$MgYj>Dux}k%k6Pf1tMRiX!5B9o3pMdUkI~b5IXx!sg;!V*_jOVoLMdv<DzZl^"
    "lNxo!LK9i1YH&#d;RNbX%Y7~VT<&3yUX^UWmvo=~`S|U{#B9Q#{fR}IgAGR5_nVNn>&N`giaRzHf765Y6E5hR?dwxI+xwTO_"
    "D<h#m97P*car*6ibozw4a9u*e+;MB(ok&fW90CJ8=j}nkot}YH{QW~HK%`^`awLN<@~3vl@YpMnM&Q1Y^U(wsle-qD#EKDj("
    "HyPKk_l@{_l5_=UJa{?&NQ<y6~QA$khW*oMi#<pg5+D3D47LFYN8~cizW-QLXO*`1?oT2Uigf&xgNzkRE#41FijFRp>R-3~p"
    "s6^$4ZOr*5FXzx=lLQEz~67LRjl;Jv<k;tN&pwU;{Jo#?voL-m;Vs%aCrUz|ma{srvI-pW4Ah<@+N8wn3QOMTl!>bkU_avb#"
    "@;{6KeI5t=dPdyC2=??aX_lD0EpRC+qck&(`;;tLNQja!|oO2d+Q>~~)SxUd$)#O{Xzid*>ZMb}PU3l@qe6JDtxuNiK<>?PU"
    "4;z?(KUaQvBKrYmQ{UW*KFQxx=Q@s@Xbih0U;lXId53~F=Mw|(%UT56v50-SuhLg&DD{7(Bzuv^Pq)x-T6Lv6=+|4F{G9lLt"
    "ywM;Rb$Q>Ud?%O(;IFuSS>Sf4BUnNetW?_>K)YwjN2T!DGa~53?I}DK1$c@2i}<m_R7n8GzPOCVLw2MeakP>gQa8We8>>$M^"
    "4eFpb!0%&QMQ!7`|6?*$r+<xj^-f6Az(1V{4dytwV3}wd!g20d*FW@Y~bbQ+9U$S>VrglD_APh2rdAev5sPdj8J?=L;q8%0u"
    "<1FU&FSFNEJXMm}h7;OkvJaQnUiUVIFBn-<i?DIP6OpNmI%rn%H-={au0)+?W~8lLZf>B<CpfljA?u6Rh{`hNnqi>lMk0Vj$"
    "_8U=5>mOR#*^r@{wo$OM0qnD^_DNRo4A8~Qu<P@>MUg|5`5LZOt$HnNa_Dq~2U$m9F@>}AL|6YMFSAUkv!AmLp*oG{>-*?h|"
    "P{G=ps7urOcZPpk0sk)kvqs=~t@-dsTC;7`KYs=v+L^qy`q``?pHh+EaTx(Wm+ogQ{aX}x&kyBi4$>QQKYg$zC*pk0)(<3nZ"
    ")ywnoQYE!$bH;^ZybR<s|Q9m`q!%lsV8{}&3_H@L{rIYXHv7I+M9Ub`ca3x#X|ZCk7BF@_8A9XrhRRX!?%|NUmYYq$pa6d7^"
    "F5lZ(aIAXpg}GVxG3_d)rA~|3c~xKc??{27b2}F=9RT`L@C?s&3*f_N@xTi(6H0=~jC2J}OKa_`cZQa8gb19}luG=-RknS2g"
    "P$snyFz?qC=BkD*|z>EuD5!w%{9MP#c-1<qm1KO6wt4F;>Pz~79*-~U66+(zo;`;lMQ`Y8Xpjy@)f*bg*`m})dNPaB#4J@mq"
    "SfV!*=>}!zzy-R+!G4px}`(~yf4<+y;pRhmUSL$KUP$##CSVlR^RqQEKETLY4vc;+kk{zs0-<&DLQ121Xy$KI+5Pjc7UGLrO"
    "ClD|4Oyu%8o8F=$*?)JE8s18g+vA?}efShUS3cn=`f(ff>Nxe3;sNyhkJCrv8|>ikVE3!2Pil(YodaK9JaYf=Ie6<6?!OHC%"
    "txcI?~%{{mwLt$^rXCn`igScf)B9Q8Pq>qM&5WR`Kx{G4?9AvmJfWlBXX`S9~fA&)eU{j&Hjpd?DIcL->6me72Ha`rWW(m0y"
    "~*W-mo`$yXV=jxSf4%9}%;PHy9TAbL(2NpW1KqGW9TTQa>^r`+PRv9dm3ykNp9Sv5%Sb>F9^Q-Gbh{68Jhj2v#mj9z=2C4)~"
    "4f*vV0FbX&4~Ju~p{rRirs1l+cezV$2UPbJ*9E<D%EQ?cxGSuW?t@Js1+nMFU7wbbFQq5qQb%V*SUwLl+7lVALrT9C2qr=7)"
    "mEW~bAg(FLV6Y^6xFq-&J{HJ;l344vAZfFR3oR8q>`%$-ZoV=lY%^`Ar3yJwx(!XF8^<oKnK+8`HyL?AKh9cCNrN}$Jldd6!"
    "duvDjbrdx}S=3ofC9foTeKqcW;(peD0DTDSMQ(R;(;wmvaA93=R3`NrEs(RLVEv-h6)y<R{T@+o{M47kBMI>JVCqTIb>z=bA"
    "8>}f6zVaR5qa;Z<K)rqpq^}F%%5W}{SzB9FURoHdX6=L<KK6&15?rWF7%%&PHbI{bs5R;DNUt!t~e(3)NRE+8R3aJ)Ol%s75"
    "n^0o#Lz1Nwh;BghN^@=Nfpg&~@<luVODZW83?Zw^H3p7xeaUL+3RvLa)<3gDbKBdMI(5d_i&Q8Wg8w#=I`F8@$m-`umKao~H"
    "%5fxF>_#(-xOKOW%sROIW_b8jMkbp`z+YT-vJ1%6z9bR>+qT+BtTr?8(?`@S_-N2o#VpWYutd@AlrK1AGXk;rke=Cs5*=lUL"
    "{wm-@K-syq&BC3JDjU-M`-G`p%QuaYT3+FMOXV|X&c?r)c|I8kTXITG${sl+Qi<>*q1EUr_hVn=B;KSd3j2@})-7oY*&W&8&"
    "!yb}v>G^*rdGGw}M--;hZw%d+Jh%V24)rkq;6JOv(Q3bGU;2IKVc(2mglk#bnD9};aYcH+LqU41U&#H4Pp<+0CY-vA{!g0Q("
    "c}QC(tD&C{-r2!ru>~`u^8(%8{C`$=ew4=i|*J3>3))aoFj<?ufqRGe;TsSFQOmkAmYM*=nHu}@>&)=D;#$*wT{Dx`){Vdi|"
    "q1Aa_?*DP2HIDYy$s{Ck~xYJ<VkDzoX#DzoyszD%L-q<GT4OxT#r!TBDfD;Y8xkkbOMs`8}BV*nRa;YCs)}`Zo-P$Ed*k48&"
    "%q=w&>Y9LHAbud0*(pUb=yp#PQp&(p*)%76YGxg3tgUrHw=hb54=0_?x)!S^~R+@9S+9aEff2kX}ayVZ$2z4TDM?)H<5lI<N"
    "!AH<n(gtzm1kCLCCh$Dr!biSJ+&jpRccBK7Kg1!jLh;z%*FH8B!`RuPhoje!6sIIgp{vn(F%D>Ssz7Smgd}__B!LL>YhYn-E"
    "<S6Q#lsA(7e+(QrlKo7L`8}p*>DS%}zUl1##h9P>3F3-O_93sKe_1Q+-5m5-af0UY3i{6;=l9fd(f42`oarEr-=FZ`FU*9GQ"
    "BGqsx&GSV(~j8dF2vFu*?-uC_)5nYfFD>(oz7-*k6qD+G4!Dw1|CWwZ{4_0#UW{Z??qp{VsIO6iK8B7PlRmWKMD6Y+PAfm8k"
    "Pps<BHd+gRj=_K~AI(T0i!uZzi5B4qqU=Jy-rWa=mNM{5;3+exCguq`>{@EcX4DqZXkweGkUsmm0-fUL=os+0R!By}6WHFv-"
    "tpaNmCX?p<;B84d6sHHhcb=V>_oUzQN>X2Lt%g&nI+95#yUOrU>{?zbDf&mev`>~8jn%%Lw{UUF%Nz_a(Hb1-1Fv-U9XJx=O"
    "rDcpM>_UCM`Kl&inrIz7$`uQ$pK2-bOie4E#`F*+8*a6AIpoGWfud>g)DsnjmJX3%=xO-xb>8g=Wzm@psUhH6x$T4OU{9Y;Q"
    "h}%*hcpdgzbuE?ILz9m_&-c-H?@H{m@a;jn-iny_lYE(+-xYz!lll0~5aF_;{O;9a;tlz0?dg3SJ2a7b9>DK>T@8<V4Y*D7r"
    "QcrEe#u=4&lyGh7KH44YdG;j)awdgH{~}hUP7;ia(@-5le~?Z3gz>D$G$4(ta>c*6tfb(mqq$}DZh21J`tNE_nqa~|E%+@A_"
    "v$oaBkzp!2Mdf@51x^CdiBU!Hf94i9PsR>6vWw+5SF}uitFsdKmnZ{J7?}5`N)o>L|5Gt2Mp$TVa#4(ck~b*$X!+kN-CQL-M"
    "LVX39Cs=SnWr(^T&VkeB>|x*W-L4msd<>=XHf-yzf<XxXQF{LYGaN%{5);DXNRoAgrr)B^0MWJ{R2QNrWpO!g3I-V4BIbzz_"
    "7GwfH&z%I4K&uV{OQFsL9eqW1Re$}f~zdQIAeT38gzaVkF<Yp$kg!;lxW50o7f%@Qo;f?#_&Rh2aHw<Gxthl3hs2zSCo~{r1"
    "hB?%sY~VMW8`EDyzip*?Q19dc)GSD!R1<fpy$WD|ALhQE=6Wr_{f|X{KKaL+*eg1d{+rW?rJiSBy>c9qgUdtTH=|yqic4nYx"
    "GgGAPbKAT>+u`NKj*j|{*}Ib`kjPd$Rmnpti=7OPo?Z)+{RvTd<BTN#aHbPyuNUt$<ma#Uuz<@c!Q}S6*sQ)YcF^c^u7#!Pr"
    "pyM0{N)IZ^h(=@3@ruL~#|ZBe!dRWI1oSJb7+kdSm*JPUXHnVQ=C+tfPEfhRH|gxa(U_^1oxj6Vr)Dlyg!Hpx;w!L;uMi(Z@"
    "XW$i9nsE&%6W%KnC$=k=o*M*n@)2|WVuF&=yQD)>gWs%GG}_KbRNsBSIp^EsP1GK=5SnoV8Y40?ps0=vb~gO23AdLW-YQvU}"
    "On04;"
)


@lru_cache(maxsize=1)
def _checkpoint_pairs():
    raw = zlib.decompress(base64.b85decode(_HEAD_DATA))
    return torch.tensor(list(struct.iter_unpack("<ff", raw)), dtype=torch.float32)


def _weights(hk, hv, rng, device, adversarial):
    # gdn.weights / gate_heads: eight extremes, two random 9B layer-0 heads,
    # then unique heads from all 4,032 entries, finally shuffle the head order.
    conv = torch.randn((2 * hk + hv) * 128, 4, generator=rng) * 0.35
    chosen = list(_EXTREME_INDICES)
    chosen.extend(
        _LAYER0_INDICES[i] for i in torch.randperm(len(_LAYER0_INDICES), generator=rng)[:2].tolist()
    )
    for index in torch.randperm(4032, generator=rng).tolist():
        if len(chosen) == hv:
            break
        if index not in chosen:
            chosen.append(index)
    chosen = torch.tensor(chosen)[torch.randperm(hv, generator=rng)]
    pairs = _checkpoint_pairs()[chosen]
    constant = torch.isin(chosen, torch.tensor(_EXTREME_INDICES[2:])).to(device)
    norm = 0.5 + 0.5 * torch.rand(128, generator=rng)
    if adversarial:
        norm[:3] = torch.tensor([0.0, 1e-3, 4.0])
    return (
        conv.to(device, torch.bfloat16),
        pairs[:, 0].to(device).contiguous(),
        pairs[:, 1].to(device).contiguous(),
        norm.to(device),
    ), constant


def _adversarial_spans(x, case, weights, rng, bulk):
    # gdn._families: draw a family then its span; never cross a sequence boundary.
    hk, hv = case.hk, case.hv
    dc, b_start = (2 * hk + hv) * 128, (2 * hk + 2 * hv) * 128
    keys = slice(hk * 128, 2 * hk * 128)
    a = slice(b_start + hv, None)
    b = slice(b_start, b_start + hv)
    dt = weights[2].double()
    A = weights[1].double().exp()
    position = 0
    for length in case.lengths:
        end = position + length
        while position < end:
            # plain, duplicate, colinear, opposite, decay-one, decay-underflow,
            # beta-zero, beta-one, scaled; robust adds zero-keys.
            family = int(torch.randint(10 if case.kind == "robust" else 9, (1,), generator=rng))
            if FORCE_FAMILY is not None:
                family = FORCE_FAMILY
            count = int(torch.randint(8, 65, (1,), generator=rng))
            if family == 4 and bool(torch.rand((), generator=rng) < 0.5):
                count = int(torch.randint(257, 1025, (1,), generator=rng))
            count = min(count, end - position)
            span = x[position : position + count]
            if family in (1, 2, 3):
                key = torch.randn(1, hk * 128, generator=bulk, device=x.device)
                if family == 1:
                    span[:, keys] = key
                elif family == 2:
                    levels = x.new_tensor([8, 16, 32])
                    indices = torch.randint(3, (count,), generator=bulk, device=x.device)
                    span[:, keys] = levels[indices, None] * key
                else:
                    signs = torch.rand(count, generator=bulk, device=x.device)
                    span[:, keys] = 0.01 * torch.where(signs < 0.5, -1.0, 1.0)[:, None] * key
            elif family == 4:
                span[:, a] = (-15 - dt).clamp(-28, 28).float()
            elif family == 5:
                decay = -100 + 74 * float(torch.rand((), generator=rng))
                target = (-decay / A).clamp(max=80)
                inverse = torch.where(target > 30, target, torch.log(torch.expm1(target)))
                span[:, a] = (inverse - dt).clamp(-28, 28).float()
            elif family in (6, 7):
                span[:, b] = -9 if family == 6 else 9
            elif family == 8:
                power = torch.randint(-10, 11, (count, 1), generator=bulk, device=x.device)
                if SCALE_PART is None:
                    span[:, :dc] *= torch.exp2(power.float())
                else:
                    lo = {"q": 0, "k": hk * 128, "v": 2 * hk * 128}[SCALE_PART]
                    hi = {"q": hk * 128, "k": 2 * hk * 128, "v": dc}[SCALE_PART]
                    span[:, lo:hi] *= torch.exp2(power.float())
            elif family == 9:
                span[:, keys] = 0
            position += count


def _initial_state(count, hv, kind, bulk, device):
    # gdn.initial_states: decayed sums of beta*v*k^T, not independent Gaussian pools.
    rank, decay = (64, 0.97) if kind == "normal" else (128, 0.995)
    shape = (count, hv, rank)
    keys = torch.randn(*shape, 128, generator=bulk, device=device)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = torch.randn(*shape, 128, generator=bulk, device=device)
    if kind == "heavy":
        values *= (0.75 * torch.randn(*shape, 1, generator=bulk, device=device)).exp()
    decay_weights = decay ** torch.arange(rank - 1, -1, -1, device=device, dtype=torch.float32)
    beta = torch.rand(*shape, generator=bulk, device=device)
    state = torch.einsum("shr,shrv,shrk->shvk", beta * decay_weights, values, keys)
    if kind in ("adversarial", "robust") and SCALE_STATE:
        state[0] = 0
        if count > 1:
            state[1] *= 1024
    return state


def make_inputs(case, seed, device="cpu"):
    """Return the ten prefill arguments, with independent CPU and device RNG streams.

    Includes eight decoy slots, fresh NaN pools, stateless sequences and BF16
    history. CPU and CUDA RNG streams differ; a seed is reproducible on the same
    device type.
    """
    if case.kind not in REGIMES or not case.lengths or min(case.lengths) < 1:
        raise ValueError("expected a supported regime and positive sequence lengths")
    if (case.hk, case.hv) not in ((8, 16), (8, 24), (16, 32), (16, 48)):
        raise ValueError("unsupported geometry")
    rng = torch.Generator().manual_seed(seed)
    bulk = torch.Generator(device=device).manual_seed(
        int(torch.randint(1 << 62, (1,), generator=rng))
    )
    adversarial = case.kind in ("adversarial", "robust")
    weights, constant = _weights(case.hk, case.hv, rng, device, adversarial)
    tokens, hv = sum(case.lengths), case.hv
    dc, b_start = (2 * case.hk + hv) * 128, (2 * case.hk + 2 * hv) * 128
    x = torch.randn(tokens, b_start + 2 * hv, generator=bulk, device=device)
    if case.kind == "heavy":
        x[:, :dc] *= (0.75 * torch.randn(tokens, dc, generator=bulk, device=device)).exp()
        channel = int(torch.randint(dc, (1,), generator=rng))
        x[: max(1, tokens // 8), channel] = 300
        x[:, b_start : b_start + hv] *= 3
    if case.kind == "heavy" or adversarial:
        # gdn._heavy_a: Student-t(3), per-head location/scale and measured clamps.
        mean = (-11 + 17 * torch.rand(hv, generator=rng)).to(device)
        spread = (0.5 + 5 * torch.rand(hv, generator=rng)).to(device)
        normal = torch.randn(tokens, hv, generator=bulk, device=device)
        chi = torch.randn(3, tokens, hv, generator=bulk, device=device).square().mean(0)
        gates = mean + spread * normal * torch.rsqrt(chi.clamp_min(1e-6))
        gates = torch.minimum(torch.maximum(gates, -15 - weights[2]), 29 - weights[2])
        x[:, b_start + hv :] = gates.clamp(-28, 28)
    if adversarial:
        _adversarial_spans(x, case, weights, rng, bulk)
    x[:, b_start + hv :] = torch.where(constant, 0, x[:, b_start + hv :])
    x[:, b_start : b_start + hv].clamp_(-9, 9)

    count = len(case.lengths)
    slots = torch.randperm(count + 8, generator=rng)[:count]
    initial = torch.ones(count, dtype=torch.bool)
    if case.kind != "heavy":
        initial = torch.rand(count, generator=rng) < 0.5
        stateless = torch.rand(count, generator=rng) < 0.2
        if bool(torch.rand((), generator=rng) < 1 / 16):
            stateless[:] = True
        slots[stateless] = -1
    state = torch.randn(count + 8, hv, 128, 128, generator=bulk, device=device)
    history = torch.randn(count + 8, 3, dc, generator=bulk, device=device)
    if adversarial:
        state.fill_(math.nan)
        history.fill_(math.nan)
    continued = slots[(slots >= 0) & initial].to(device)
    if continued.numel():
        state[continued] = _initial_state(len(continued), hv, case.kind, bulk, device)
        history[continued] = (
            torch.randn(len(continued), 3, dc, generator=bulk, device=device).bfloat16().float()
        )
    fresh = slots[(slots >= 0) & ~initial].to(device)
    state[fresh], history[fresh] = math.nan, math.nan
    cu = torch.tensor((0, *case.lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return (
        x.bfloat16(),
        *weights,
        cu.to(device),
        slots.to(device, torch.int32),
        initial.to(device),
        state.contiguous(),
        history.bfloat16().contiguous(),
    )
