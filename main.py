import random

clock = 1

class Customer:
    def __init__(self, arrival_time, service_start_time, service_time):
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_time = service_time
        self.service_end = -1
        self.service_wait = -1

    def end(self, time):
        self.service_end = time
        self.service_wait = time - self.arrival_time - self.service_time


class PreProcessor1:
    def __init__(self, lamb, mu, k):
        self.lamb = lamb
        self.mu1 = mu
        self.queue = [None] * k

    def simulation(self, time_passed, time):
        if len(self.queue) == 0:
            time = time + time_passed
            temp_time = 0
            count = 0
            while temp_time < clock:
                next_customer = random.expovariate(self.lamb)
                temp_time  = temp_time + next_customer
                if temp_time < clock:
                    count = count + 1
            for i in range(count):

        else:


        return time, next_customer



def simulation(lambda1, lambda2, mu1, mu2, m3, sim_time):
    # universal clock
    time = 0

    # processor queues
    pre_processor_1_queue = []
    pre_processor_2_queue = []
    main_processor_queue = []

    # ----------------------------------
    # The actual simulation happens here:
    while t < simulation_time:

        # calculate arrival date and service time for new customer
        if len(Customers) == 0:
            arrival_date = neg_exp(lambd)
            service_start_date = arrival_date
        else:
            arrival_date += neg_exp(lambd)
            service_start_date = max(arrival_date, Customers[-1].service_end_date)
        service_time = neg_exp(mu)

        # create new customer
        Customers.append(Customer(arrival_date, service_start_date, service_time))

        # increment clock till next end of service
        t = arrival_date
    # ----------------------------------

    # calculate summary statistics
    Waits = [a.wait for a in Customers]
    Mean_Wait = sum(Waits) / len(Waits)

    Total_Times = [a.wait + a.service_time for a in Customers]
    Mean_Time = sum(Total_Times) / len(Total_Times)

    Service_Times = [a.service_time for a in Customers]
    Mean_Service_Time = sum(Service_Times) / len(Service_Times)

    Utilisation = sum(Service_Times) / t

    # output summary statistics to screen
    print()
    ""
    print
    "Summary results:"
    print
    ""
    print
    "Number of customers: ", len(Customers)
    print
    "Mean Service Time: ", Mean_Service_Time
    print
    "Mean Wait: ", Mean_Wait
    print
    "Mean Time in System: ", Mean_Time
    print
    "Utilisation: ", Utilisation
    print
    ""

    # prompt user to output full data set to csv
    if input("Output data to csv (True/False)? "):
        outfile = open('MM1Q-output-(%s,%s,%s).csv' % (lambd, mu, simulation_time), 'wb')
        output = csv.writer(outfile)
        output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date'])
        i = 0
        for customer in Customers:
            i = i + 1
            outrow = []
            outrow.append(i)
            outrow.append(customer.arrival_date)
            outrow.append(customer.wait)
            outrow.append(customer.service_start_date)
            outrow.append(customer.service_time)
            outrow.append(customer.service_end_date)
            output.writerow(outrow)
        outfile.close()
    print
    ""
    return
