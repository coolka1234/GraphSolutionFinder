from djikstra import run_djikstra, test_run_djikstra
from a_star import run_a_star_time, run_a_star_line 
from tabu_playground import run_tabu_time, run_tabu_line 
def main():
    print("Welcome to the Public Transport Route Planner!")
    print("Please select an option:")
    print("1. Djikstra")
    print("2. A*")
    print("3. Tabu Search")
    print("4. Exit")
    option=input("Enter your choice: ")
    if option == '1':
        djikstra()
    elif option == '2':
        tabu_search()
    elif option == '3':
        tabu_search()
    elif option == '4':
        exit()
    else:
        print("Invalid choice. Please try again.")
        main()

    def tabu_search():
        start_stop = input("Podaj przystanek początkowy: ")
        
        stops_to_visit = input("Podaj przystanki do odwiedzenia (oddzielone średnikiem): ").split(";")
        
        optimization_criterion = input("Podaj kryterium optymalizacyjne (t - czas, p - liczba zmian linii): ")
        
        start_time = input("Podaj czas pojawienia się na przystanku początkowym: ")
        
        print("\nWczytane dane:")
        print(f"Przystanek początkowy: {start_stop}")
        print(f"Przystanki do odwiedzenia: {', '.join(stops_to_visit)}")
        print(f"Kryterium optymalizacyjne: {optimization_criterion}")
        print(f"Czas pojawienia się na przystanku początkowym: {start_time}")

        if optimization_criterion == 't':
            run_tabu_time(start_stop, stops_to_visit, start_time)
        elif optimization_criterion == 'p':
            run_tabu_line(start_stop, stops_to_visit, start_time)
        else:
            print("Invalid criterion. Please try again.")
            tabu_search()
    
    def djikstra():
        start_stop = input("Podaj przystanek początkowy: ")
        
        end_stop = input("Podaj przystanek końcowy: ")

        start_time=input("Podaj czas pojawienia się na przystanku początkowym: ")
        
        print("\nWczytane dane:")
        print(f"Przystanek początkowy: {start_stop}")
        print(f"Przystanek końcowy: {end_stop}")
        print(f"Czas pojawienia się na przystanku początkowym: {start_time}")
        run_djikstra(start_stop, end_stop, start_time)
        # test_run_djikstra()
    
    def a_star():
        start_stop = input("Podaj przystanek początkowy: ")

        end_stop = input("Podaj przystanek końcowy: ")
        
        start_time = input("Podaj czas pojawienia się na przystanku początkowym: ")
        
        criteria=input("Podaj kryterium optymalizacyjne (t - czas, p - liczba zmian linii): ")

        print("\nWczytane dane:")
        print(f"Przystanek początkowy: {start_stop}")
        print(f"Przystanek końcowy: {end_stop}")
        print(f"Czas pojawienia się na przystanku początkowym: {start_time}")
        print(f"Kryterium optymalizacyjne: {criteria}")
        if criteria == 't':
            run_a_star_time(start_stop, end_stop, start_time)
        elif criteria == 'p':
            run_a_star_line(start_stop, end_stop, start_time)
        else:
            print("Invalid criteria. Please try again.")
            a_star()

        
    

if __name__ == "__main__":
    main()
