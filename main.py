import numpy

clock = 1  # base unit time


class Customer:
    def __init__(self, arrival_time, service_start_time, mu):
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_time = abs(numpy.random.normal(1 / mu))
        self.service_end = -1
        self.service_wait = -1

    def end(self, time):
        self.service_end = time
        self.service_wait = time - self.arrival_time - self.service_time


class PreProcessor1:
    def __init__(self, lamb, k):
        self.lamb = lamb
        self.queue = [None] * k
        self.k = k

    def simulation(self, time_passed, time):
        # done tasks
        done = []

        c_count = 0  # customers count
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if c_count == 0:
            # manage the queue
            time = time + time_passed
            next_customers = numpy.random.poisson(self.lamb)
            if next_customers > self.k:
                for i in range(self.k):
                    self.queue[i] = Customer(time, 0, 5)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, 0, 5)
        else:
            # processing time
            p_time = 0

            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find shortest customer task
                min_s_time = 1000
                for c in self.queue:
                    if c is not None:
                        if c.service_time < min_s_time:
                            min_s_time = c.service_time
                            c_min_index = self.queue.index(c)

                # process
                if min_s_time + p_time > clock:
                    self.queue[c_min_index].service_time = min_s_time - (clock - p_time)
                    self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    p_time = clock
                else:
                    self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_end = time + time_passed
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    done.append(self.queue[c_min_index])
                    self.queue[c_min_index] = None
                    c_count = c_count - 1
                    p_time = p_time + min_s_time

            # manage the queue
            time = time + time_passed
            if len(self.queue) == 0:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers > self.k:
                    for i in range(self.k):
                        self.queue[i] = Customer(time, 0, 5)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, 0, 5)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, 0, 5)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, 0, 5)
                                temp = temp - 1

        return done


class PreProcessor2:
    def __init__(self, lamb, k):
        self.lamb = lamb
        self.queue = [None] * k
        self.k = k

    def simulation(self, time_passed, time):
        # done tasks
        done = []

        c_count = 0  # customers count
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if c_count == 0:
            # manage the queue
            time = time + time_passed
            next_customers = numpy.random.poisson(self.lamb)
            if next_customers > self.k:
                for i in range(self.k):
                    self.queue[i] = Customer(time, 0, 5)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, 0, 5)
        else:
            # processing time
            p_time = 0
            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find a random not none index
                found = False
                while not found:
                    index = numpy.random.randint(0, self.k - 1)
                    if self.queue[index] is not None:
                        found = True

                # process
                if self.queue[index].service_time + p_time > clock:
                    self.queue[index].service_time = self.queue[index].service_time - (clock - p_time)
                    self.queue[index].service_start_time = time
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    p_time = 1
                else:
                    self.queue[index].service_start_time = time
                    self.queue[index].service_end = time + time_passed
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    done.append(self.queue[index])
                    self.queue[index] = None
                    c_count = c_count - 1
                    p_time = p_time + self.queue[index].service_time

            # manage the queue
            time = time + time_passed
            if len(self.queue) == 0:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers > self.k:
                    for i in range(self.k):
                        self.queue[i] = Customer(time, 0, 3)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, 0, 3)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, 0, 3)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, 0, 3)
                                temp = temp - 1

        return done


class MainProcessor:
    def __init__(self, k):
        self.queue = [None] * k
        self.k = k
        self.done = []

    def simulation(self, time_passed, time, next_customers):
        c_count = 0  # customers count
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if len(next_customers) == 0 and c_count == 0:
            time = time + time_passed

        else:
            # put new arrival in main processor queue (left-overs dropped!)
            for c_a in next_customers:
                for c in self.queue:
                    if c is None:
                        self.queue[next_customers.index(c)] = c_a
                        c_count = c_count + 1
                        break
                if c_count == self.k:
                    break

                # process
                for c in self.queue:
                    if c is not None:
                        if c.service_time - (clock/c_count) <= 0:
                            index = self.queue.index(c)
                            self.queue[index].service_start_time = time
                            self.queue[index].service_end = time + time_passed
                            self.queue[index].service_wait = 0
                            self.done.append(c)
                            self.queue[index] = None
                        else:
                            self.queue[index].service_time = self.queue[index].service_time - (clock/c_count)
                            self.queue[index].service_start_time = time
                            self.queue[index].service_wait = 0

        return time


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
            arrival_date = (lambd)
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
