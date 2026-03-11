# Initial data converter for the Nathan trajectory simulator (EXCEL)

<img width="890" height="542" alt="image" src="https://github.com/user-attachments/assets/0a0f010a-2b4d-4c72-abec-aadf0a7f4566" />

Alan NathanはEXCELで投球や打球の軌道シミュレータを作り，下記に公開しています．

ここで公開している，初期値変換コード（rapsodo_to_nathan.py）は，そのシミュレータに入力する，角速度データの値の計算を補助するためのツールです．一般にはRapsodoなどで，出力される角速度（回転寿）データと，Nathanのシミュレータの角速度データとの整合性がないので，それを補助する目的で作りました．

ただ，ひとつだけお許しいただきたいことがあります．Rapsodoが示す「回転方向の角度」（投手から見た回転軸の水平面になす角度）は，時刻表記と物理的な表記となじまない（不適切な）角度表記のため，この角度は，ご自身で適切な角度（deg）に手動で変換していただきくことをお願いしております．ご不便をおかけしますが何卒よろしくお願いいたします．

## Nathanの軌道シミュレータの概要
野球の物理の研究の第一人者のNathanは，the physics of baseballというページを作り，そこに様々な情報を公開しています．

このページのうち，[Trajectory Calculator](https://baseball.physics.illinois.edu/trajectory-calculator-new3D.html)
に置いてある，EXCELで書かれたファイルBaseball Trajectory [Calculator--new 3D version: updated, November 13, 2021](https://baseball.physics.illinois.edu/TrajectoryCalculator-new-3D-May2021.xlsx) を使用すると，環境変数（気温や高度など）とボールの運動学的な初期値（初速度の大きさ，方向，角速度）を入力すると，ピッチングマウンドからホームベースまでの軌道を計算できます．もともとはバッティングの軌道計算だけでしたが，新しく投球軌道シミュレータのEXCELのタブを追加しています．

内容を確認しましたが，計算方法は非常に丁寧です．Alan Nathanは物理学者なので，非常に丁寧な仕事をされています．ただしEXCELなので積分方法は単純です．私はPythonコードでルンゲ・クッタ法に書き換えていますが，この程度の計算では大きな違いは発生しません．

単位はアメリカのft, mphなど異なるので注意が必要ですが，EXCELの必要なところに，単位を変換して入力すれば，X, Y, Zの軌道を計算できます．

ただし一番厄介なことは，
### 「ボールの角速度ベクトルの初期境界条件の入力」
です，多くの人はRapsodoなどの計測機器による指標に慣れていると思いますが，Nathanのシミュレータの初期値の定義は少し異なり，RapsodoやStatcastから出力される結果をそのまま使えません．

Rapsodoなどは，角速度の大きさ（回転数：rpm）と，回転軸の水平角度（回転方向：しかも角度ではなく時刻表記）と方位角（ジャイロ角度）を与えています．
これに対してNathanは，角度ではなく，**バックスピンの回転数**と，**サイドスピンの回転数**という数値を初期条件として使用しています．

この変換は，単位だけの変換では計算できません．そこで，
### Pythonで計算する回転数（角速度）パラメータ変換コード（rapsodo_to_nathan.py）
を作りました．角速度（回転数），初速度などのデータを入力し，Pythonコードで計算をすることで，ターミナルに「バックスピンの回転数」と，「サイドスピンの回転数」を含めて，初期境界条件パラメータを出力します．

Pythonコードに，Rapsodoなどで得られた，角速度データ（回転数データ）を入力し，コードを実行してください．すると，ターミナルの出力に，EXCELで必要とされる，角速度の「BackSpin (rpn)」成分と「SideSpin (rpm）」成分を計算し出力します．

Nathanの計算方法の詳細は，[論文: Analysis of Baseball Trajectories](https://baseball.physics.illinois.edu/TrajectoryAnalysis.pdf)と，EXCELの最後のタブにあるReadMeなどを御覧ください．

## シミュレータの座標系
X：左右方向：３塁ー＞１塁：正
Y：捕手ー＞投手方向：正
Z：鉛直上方：正

## 初期値変換コード（rapsodo_to_nathan.py）使用方法

このコードは，コード内で

入力として，

        １．「最高速度」＝初期速度 (km/h)：　例，v0_mps=132.7,  # （初期速度と同一と解釈，km/hで入力）
        
        ２．「縦のリリース角度」＝初速度の上下角度 (deg) ：　例，vel_angle_vertical_deg=-1.47, # ボールの上下速度方向角度（下向きが正なので注意）
        
        ３．「横のリリース角度」＝初速度の左右角度 (deg)：　例，vel_azimuth_deg=2.0, 　方位角　（例：右ピッチャーがマウンドの右側から，ホームベースの内側の左方向に投げるときは正）
        
        ４．「回転数」＝角速度のノルム　(回転数，rpm)：　例，spin_rate_rpm=2152.3,
        
        ５．「回転方向」＝　投手から見た回転軸の水平面になす角度（deg）：　例，spin_tilt_deg=-25.0, 投手から見た回転軸の水平軸に対する角度（ユーザが角度に変換してほしい。ここでは0としています）
            回転軸が水平面となす角 [deg]。0＝水平、90＝鉛直（後方から見た傾き）
            通常，右投手のストレートの場合：水平面から下向きになるので，数値は通常は負の数値となる
            **変換コードを作っていません．Rapsodoの時間表記がナンセンスで嫌いなだけです．気が向いたら変換できるようにします．**

        ６．「ジャイロ角度」＝回転軸の方位（真上から見た）（deg）：例，spin_azimuth_deg=0.0
            例：spin_azimuth_deg 0＝捕手方向（-Y）と一致

を与えて（書き換えて），Pythonで実行します．
このファイルの中身を書き換えてください．最後のmain関数の部分です．
実行例：
        
        cd ◯◯/◯◯/ # このPythonコードのあるフォルダー（ディレクトリ）に移動
        
        python3 rapsodo_to_nathan.py

<img width="1362" height="464" alt="image" src="https://github.com/user-attachments/assets/003927c4-51b9-49b0-9369-00c09cf951d9" />

角度の単位の定義などは，Pythonコードを御覧ください．

## Rapsodoの表示例
<img width="2045" height="1345" alt="スクリーンショット 2026-03-11 18 38 14" src="https://github.com/user-attachments/assets/68e2b387-2233-4343-bb19-227f1be38f3f" />

図は，[ラプソード計測データ解説①「球速」](https://note-rapsodojp.rapsodo.com/n/n8ad1ed6f0109)　より引用しました．

## NathanのPitchedBallTrajectoryの入力例
EXCELの左側の赤色の枠線で囲ってある部分を，このシミュレーションではいじっています．

さきほどの**rapsodo_to_nathan.py**は，このうち黄色い部部分の，初期値を計算します．release speed (mph)，は単位を変えただけ，release angle (deg)とrelease direction (deg)はそのまま入力すればよいのですが，これらの値を，backspin (rpm)やsidespin (rpm)の，角速度ベクトルの初期入力値の計算に使用します．

コードやこの説明に不備があるかもしれませんが，その場合，ご容赦ください．不備のご指摘に関しては，お手数ですが，[SkillVis](skill-vs.com)までに，お問い合わせください．

### なお，不備のご指摘やご要望をいただいいた場合，内容は拝見しますが，返信を行わない予定ですので，あらかじめご承知ください．申し訳ありません．


<img width="1256" height="892" alt="image" src="https://github.com/user-attachments/assets/d3f1aff9-0de8-435f-a46b-8a5f72c73389" />
