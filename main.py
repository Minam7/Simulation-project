import numpy
import csv

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
                    self.queue[i] = Customer(time, -1, 5)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, -1, 5)
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
                    if self.queue[c_min_index].service_start_time == -1:
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
                        self.queue[i] = Customer(time, -1, 5)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, -1, 5)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, -1, 5)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, -1, 5)
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
                    self.queue[i] = Customer(time, -1, 3)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, -1, 3)
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
                    index = numpy.random.randint(0, self.k)
                    if self.queue[index] is not None:
                        found = True

                # process
                if self.queue[index].service_time + p_time > clock:
                    self.queue[index].service_time = self.queue[index].service_time - (clock - p_time)
                    self.queue[index].service_start_time = time
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    p_time = 1
                else:
                    if self.queue[index].service_start_time == -1:
                        self.queue[index].service_start_time = time
                    self.queue[index].service_end = time + time_passed
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    done.append(self.queue[index])
                    p_time = p_time + self.queue[index].service_time
                    self.queue[index] = None
                    c_count = c_count - 1

            # manage the queue
            time = time + time_passed
            if len(self.queue) == 0:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers > self.k:
                    for i in range(self.k):
                        self.queue[i] = Customer(time, -1, 3)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, -1, 3)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, -1, 3)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, -1, 3)
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
                        c_a.service_time = abs(numpy.random.normal(1 / 1))
                        self.queue[self.queue.index(c)] = c_a
                        c_count = c_count + 1
                        break
                if c_count == self.k:
                    break

            # process
            for c in self.queue:
                if c is not None:
                    if c.service_time - (clock / c_count) <= 0:
                        index = self.queue.index(c)
                        self.queue[index].service_start_time = time
                        self.queue[index].service_end = time + time_passed
                        self.done.append(c)
                        self.queue[index] = None
                    else:
                        index = self.queue.index(c)
                        self.queue[index].service_time = self.queue[index].service_time - (clock / c_count)
                        self.queue[index].service_start_time = time

        return None


class MainProcessorExtra:
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
                        c_a.service_time = abs(numpy.random.normal(1 / 1))
                        self.queue[self.queue.index(c)] = c_a
                        c_count = c_count + 1
                        break
                if c_count == self.k:
                    break

            # process section
            p_time = 0  # processing time
            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find shortest customer arrival time
                min_a_time = 10000000000000000000
                for c in self.queue:
                    if c is not None:
                        if c.arrival_time < min_a_time:
                            min_a_time = c.arrival_time
                            c_min_index = self.queue.index(c)

                # process
                if self.queue[c_min_index].service_time + p_time > clock:
                    self.queue[c_min_index].service_time = self.queue[c_min_index].service_time - (clock - p_time)
                    self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    p_time = clock
                else:
                    if self.queue[c_min_index].service_start_time == -1:
                        self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_end = time + time_passed
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    self.done.append(self.queue[c_min_index])
                    p_time = p_time + self.queue[c_min_index].service_time
                    self.queue[c_min_index] = None
                    c_count = c_count - 1
                time = time + time_passed

        return None


def processor_sharing():
    pp1 = PreProcessor1(7, 100)
    pp2 = PreProcessor2(2, 12)
    mp = MainProcessor(8)
    time = 0
    simulation_times = 5000000
    while len(mp.done) < simulation_times:
        output1 = pp1.simulation(clock, time)
        output2 = pp2.simulation(clock, time)
        output = output1 + output2
        mp.simulation(clock, time, output)
        time = time + clock

    # calculate summary statistics
    waits = [c.service_wait for c in mp.done]
    mean__wait = sum(waits) / len(waits)
    print("Mean Wait : ")
    print(mean__wait)

    total__times = [c.service_wait + c.service_time for c in mp.done]
    mean__time = sum(total__times) / len(total__times)
    print("Total Times : ")
    print(mean__time)

    service__times = [c.service_time for c in mp.done]
    mean__service__time = sum(service__times) / len(service__times)
    print("Service Times : ")
    print(mean__service__time)

    utilisation = sum(service__times) / time
    print("Utilisation : ")
    print(utilisation)

    # write output full data set to csv
    outfile = open('system_output_%s_simulations.csv' % simulation_times, 'w')
    output = csv.writer(outfile)
    output.writerow(
        ['Customer', 'Arrival_Time', 'Service_Start_Time', 'Service_Time', 'Service_Wait', 'Service_End'])
    i = 0
    for customer in mp.done:
        i = i + 1
        outrow = []
        outrow.append(i)
        outrow.append(customer.arrival_time)
        outrow.append(customer.service_start_time)
        outrow.append(customer.service_time)
        outrow.append(customer.service_wait)
        outrow.append(customer.service_end)
        output.writerow(outrow)
    outfile.close()


def first_come_first_served():
    # Extra Section
    print('Extra Section Started')
    pp1 = PreProcessor1(7, 100)
    pp2 = PreProcessor2(2, 12)
    mp = MainProcessorExtra(8)
    time = 0
    simulation_times = 5000000
    while len(mp.done) < simulation_times:
        output1 = pp1.simulation(clock, time)
        output2 = pp2.simulation(clock, time)
        output = output1 + output2
        mp.simulation(clock, time, output)
        time = time + clock

    # calculate summary statistics
    waits = [c.service_wait for c in mp.done]
    mean__wait = sum(waits) / len(waits)
    print("Mean Wait : ")
    print(mean__wait)

    total__times = [c.service_wait + c.service_time for c in mp.done]
    mean__time = sum(total__times) / len(total__times)
    print("Total Times : ")
    print(mean__time)

    service__times = [c.service_time for c in mp.done]
    mean__service__time = sum(service__times) / len(service__times)
    print("Service Times : ")
    print(mean__service__time)

    utilisation = sum(service__times) / time
    print("Utilisation : ")
    print(utilisation)

    # write output full data set to csv
    outfile = open('system_output_%s_simulations_extra.csv' % simulation_times, 'w')
    output = csv.writer(outfile)
    output.writerow(
        ['Customer', 'Arrival_Time', 'Service_Start_Time', 'Service_Time', 'Service_Wait', 'Service_End'])
    i = 0
    for customer in mp.done:
        i = i + 1
        outrow = []
        outrow.append(i)
        outrow.append(customer.arrival_time)
        outrow.append(customer.service_start_time)
        outrow.append(customer.service_time)
        outrow.append(customer.service_wait)
        outrow.append(customer.service_end)
        output.writerow(outrow)
    outfile.close()


if __name__ == '__main__':
    Extra = True
    Normal = True
    if Normal:
        processor_sharing()
    if Extra:
        first_come_first_served()
