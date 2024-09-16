from omegaconf import OmegaConf
from codes.boot_mode import boot_web
from codes.boot_mode import boot_cmd


if __name__ == "__main__":
    print("起動モードを選んでください。")
    print("1:コマンド、2:Web")

    # 入力を取得
    choice = input("選択肢 (1または2): ")

    if choice == "1":
        print("コマンドモードで起動します...")
        boot_cmd()  # コマンドラインモードの処理を実行

    elif choice == "2":
        print("Webモードで起動します...")
        boot_web()  # Webモードの処理を実行

    else:
        print("無効な選択肢です。1または2を選んでください。")
