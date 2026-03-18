# Initial condition converter for the Nathan trajectory simulator (Rapsodo to EXCEL)

2026.03.18訂正（様々な不備に対応し，Rapsodo形式の回転角度の時計入力，リリース位置の変換，左右投手の区別などを追加）

---
野球の研究を行っている，物理学者Alan NathanはEXCELで投球や打球の軌道シミュレータを作り，下記に公開しています．

ここで公開している，初期値変換コード（rapsodo_to_nathan.py）は，そのシミュレータに入力する，角速度データの値の計算を補助するためのツールです．一般にはRapsodoで，出力される角速度（回転寿）データと，Nathanのシミュレータの角速度データとの整合性がないので，それを補助する目的で作りました．

EXCELで作られたシミュレータで使用している単位系はアメリカ仕様でftなどを用いています．もしmなどに変換が必要な場合は，新たにセルを作って変換すると良いでしょう．

１ [ft] = 0.3048 [m] です．

## Nathanの軌道シミュレータの概要
<img width="890" height="542" alt="image" src="https://github.com/user-attachments/assets/0a0f010a-2b4d-4c72-abec-aadf0a7f4566" />


野球の物理の研究の第一人者のNathanは，the physics of baseballというページを作り，そこに様々な情報を公開しています．

このページのうち，[Trajectory Calculator](https://baseball.physics.illinois.edu/trajectory-calculator-new3D.html)
に置いてある，EXCELで書かれたファイルBaseball Trajectory [Calculator--new 3D version: updated, November 13, 2021](https://baseball.physics.illinois.edu/TrajectoryCalculator-new-3D-May2021.xlsx) を使用すると，環境変数（気温や高度など）とボールの運動学的な初期値（初速度の大きさ，方向，角速度）を入力すると，ピッチングマウンドからホームベースまでの軌道を計算できます．もともとはバッティングの軌道計算だけでしたが，新しく投球軌道シミュレータのEXCELのシート（PitchedBallTrajectory）を追加しています．

Alan Nathanは物理学者なので，非常に丁寧な仕事をされており，このシミュレータの計算方法は非常に丁寧に記述されています．ただしEXCELなので積分方法は単純です．NathanのEXCELコードをPythonに変換（未公開）し，ルンゲ・クッタ法による積分に書き換えてもいますが大きな違いは発生しません．

単位はアメリカのft, mphなど異なるので注意が必要ですが，EXCELの必要なところに，単位を変換して入力すれば，X, Y, Zの軌道を計算できます．

ただし一番厄介なことは，
### 「ボールの角速度ベクトルの初期境界条件の入力」
です，多くの人はRapsodoなどの計測機器による指標に慣れていると思いますが，Nathanのシミュレータの初期値の定義は少し異なり，RapsodoやStatcastから出力される結果をそのまま使えません．

Rapsodoなどは，角速度の大きさ（回転数：rpm）と，回転軸の水平角度（回転方向：角度ではなく時刻表記）と方位角（ジャイロ角度）を与えています．
これに対してNathanは，角度ではなく，**バックスピンの回転数**と，**サイドスピンの回転数**という数値を初期条件として使用しています．

この変換は，単位だけの変換では計算できません．そこで，
### Pythonで計算する回転数（角速度）パラメータ変換コード（rapsodo_to_nathan.py）
を作りました．角速度（回転数），初速度などのデータを入力し，Pythonコードで計算をすることで，ターミナルに「バックスピンの回転数」と，「サイドスピンの回転数」を含めて，初期境界条件パラメータを出力します．

Pythonコードに，Rapsodoなどで得られた，角速度データ（回転数データ）を入力し，コードを実行してください．すると，ターミナルの出力に，EXCELで必要とされる，角速度の「BackSpin (rpn)」成分と「SideSpin (rpm）」成分を計算し出力します．

Nathanの計算方法の詳細は，[論文: Analysis of Baseball Trajectories](https://baseball.physics.illinois.edu/TrajectoryAnalysis.pdf)と，EXCELの最後のタブにあるReadMeなどを御覧ください．

## シミュレータの座標系
<img width="339" height="251" alt="image" src="https://github.com/user-attachments/assets/f43846d1-45d9-49a8-8182-282732589c46" />

X：左右方向：３塁ー＞１塁：正

Y：捕手ー＞投手方向：正

Z：鉛直上方：正

<img width="1327" height="1430" alt="nathan_vectors_illustrator_v2" src="https://github.com/user-attachments/assets/d5224878-7737-48fa-a970-14cc52797ce3" />

角速度の座標系は，ボールの初速度方向（法線方向）とそれに直交する面で構成されています．

## Rapsodoの表示例
![Rapsodo_#2](https://github.com/user-attachments/assets/45dc543a-2473-4b64-9566-bd07931b3a91)


図は，[ラプソード計測データ解説①「球速」](https://note-rapsodojp.rapsodo.com/n/n8ad1ed6f0109)　より引用しました．

次の図はRapsodoが定義していると思われる定義を示した図です．XYZの座標系はNathanの定義を踏襲しただけです．Rapsodoには座標系の記述がありません．

また，角度などには言葉だけの定義が多く，厳密な定義がないため，この変換コードは言葉からの推測により式を作成しております．

<img width="1387" height="1515" alt="Rapsodo" src="https://github.com/user-attachments/assets/efc828be-2e68-49d9-8ef6-ab3364777ff8" />

## 初期値変換コード（rapsodo_to_nathan.py）使用方法

このコードは，コード内で

入力として，

        １．「最高速度」＝初期速度 (km/h)：　例，v0_mps=132.7,  # （初期速度と同一と解釈，km/hで入力）
        
        ２．「縦のリリース角度」＝初速度の上下角度 (deg) ：　例，vel_angle_vertical_deg=-1.47, # ボールの上下速度方向角度（下向きが正なので注意）
        
        ３．「横のリリース角度」＝初速度の左右角度 (deg)：　例，vel_azimuth_deg=2.0, 　方位角　（例：右ピッチャーがマウンドの右側から，ホームベースの内側の左方向に投げるときは正）
        
        ４．「回転数」＝角速度のノルム　(回転数，rpm)：　例，spin_rate_rpm=2152.3,
        
        ５．「回転方向」＝　"HH:MM"の時刻で入力（Rapsodoの出力が，「01:18」なら「"01:18"」のように，ダブルクォーテーションではさみ文字列として記述してください．
        Nathanのシミュレータの定義は投手から見た回転軸の水平面になす角度（deg）に変換します．
        例，spin_tilt_deg="01:18", 投手から見た回転軸の水平軸に対する角度の時刻表期
            回転軸が水平面となす角 [deg]。0＝水平、90＝鉛直（後方から見た傾き）
            通常，右投手のストレートの場合：水平面から下向きになるので，数値は通常は負の数値となる

        ６．「ジャイロ角度」＝回転軸の方位（真上から見た）（deg）：例，spin_azimuth_deg=0.0
            例：spin_azimuth_deg 0＝捕手方向（-Y）と一致

を与えて（書き換えて），Pythonで実行します．これらの計算には投手の利き腕が考慮さされて変換されるものもあります．

このファイルの下の図のハイライトされた部分の中身を書き換えてください．最後のmain関数の部分です．
実行例：
        
        cd ◯◯/◯◯/code # このPythonコードのあるフォルダー（ディレクトリ）に移動
        
        python3 rapsodo_to_nathan.py

<img width="1186" height="694" alt="image" src="https://github.com/user-attachments/assets/10e543a4-c62f-4d60-8de1-ad22987c9792" />

角度の単位の定義などは，Pythonコードを御覧ください．

ターミナルの出力例

        % python3 rapsodo_to_nathan.py

        PitchParameters: PitchParameters(x0=0.47, y0=16.764, z0=1.5, v0_mps=37.611111111111114, theta_deg=0.1, phi_deg=2.6, backspin_rpm=1062.736282989416, sidespin_rpm=-1377.8913828343998, wg_rpm=450.712026436565, batter_hand='R')
        backspin_rpm=1062.7, sidespin_rpm=-1377.9, wg_rpm=450.7
        --- Nathan Excel 用 ---
        辞書: {'release_speed_mph': 84.13365942893493, 'release_angle_deg': 0.1, 'release_direction_deg': 2.6, 'backspin_rpm': 1062.736282989416, 'sidespin_rpm': -1377.8913828343998, 'wg_rpm': 450.712026436565, 'x0_ft': 1.5419947999999999, 'y0_ft': 55.000001759999996, 'z0_ft': 4.92126, 'x0_m': 0.47, 'y0_m': 16.764, 'z0_m': 1.5}
        貼り付け用1行: 84.1337  0.1000  2.6000  1062.74 -1377.89        450.71  1.5420  55.0000 4.9213

ターミナル出力の「辞書」または「貼り付け用1行」のデータを以下のNathanのEXCELのシミュレータPitchedBallTrajectoryの相当するセルにコピペしてください．

## NathanのPitchedBallTrajectoryの入力例
EXCELの左側の赤色の枠線で囲ってある（黃色にハイライトした）部分を，このシミュレーションではいじっています．

さきほどの**rapsodo_to_nathan.py**は，このうち黄色い部部分の，初期値を計算します．release speed (mph)，は単位を変えただけ，release angle (deg)とrelease direction (deg)はそのまま入力すればよいのですが，これらの値を，backspin (rpm)やsidespin (rpm)の，角速度ベクトルの初期入力値の計算に使用します．

Nathanの計算結果は，アメリカの単位系[ft]などを使用しています．必要に応じて，ご自身で[m]などに変換してください．**赤枠の入力部分はオリジナルの単位系で入力をお願いします．**

コードやこの説明に不備があるかもしれませんが，その場合，ご容赦ください．不備のご指摘に関しては，お手数ですが，[SkillVis](skill-vs.com)までに，お問い合わせください．

<img width="1434" height="857" alt="image" src="https://github.com/user-attachments/assets/04f766ef-3b46-42d5-a1ea-e2ec1c707305" />

それ以外の，青色で示された環境変数（標高，温度，風速，気圧等）も変更可能です．軌道計算に大きな影響は与えないので，そのままでもよいと思います．

### 免責
不備のご指摘やご要望をいただいいた場合，返信を行わないこともあるかもしれませんが，あらかじめご承知ください．申し訳ありません．

プログラム等には不備を含む場合があります．本内容を参照して研究などを行う場合には，その点を十分に踏まえた上で，自己責任でご利用ください．また，掲載内容によって生じた一切の損害については，著者はその責を負わないものとします．

