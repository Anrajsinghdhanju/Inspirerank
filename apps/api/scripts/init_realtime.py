from app.services.realtime_feedback import get_feedback_store


def main() -> None:
    store = get_feedback_store()
    store.ensure_table()
    print("Realtime interaction table is ready.")


if __name__ == "__main__":
    main()
