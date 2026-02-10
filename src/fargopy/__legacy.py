def load_allfields(self, fluid, snapshot=None, type="scalar"):
        """
        Load all fields in the output directory for a given fluid.

        Parameters
        ----------
        fluid : str
            Name of the fluid (e.g., 'gas').
        snapshot : int, optional
            Snapshot index to load. If None, loads all snapshots.
        type : str, optional
            Field type ('scalar' or 'vector').

        Returns
        -------
        fargopy.Dictobj
            Object containing all loaded fields.
        """
        qall = False
        if snapshot is None:
            qall = True
            fields = fargopy.Dictobj()
        else:
            fields = fargopy.Dictobj()

        # Search for field files
        pattern = os.path.join(self.output_dir, f"{fluid}*.dat")
        import glob

        files_found = sorted(glob.glob(pattern))

        if files_found:
            size = 0
            for file_field in files_found:
                comps = Simulation._parse_file_field(file_field)
                if comps:
                    if qall:
                        # Store all snapshots
                        field_name = comps[0]
                        field_snap = int(comps[1])

                        if type == "scalar":
                            field_data = self._load_field_scalar(file_field)
                        elif type == "vector":
                            field_data = []
                            variables = ["x", "y"]
                            if self.vars.DIM == 3:
                                variables += ["z"]
                            for i, variable in enumerate(variables):
                                file_name = f"{fluid}{variable}{str(field_snap)}.dat"
                                file_field = os.path.join(self.output_dir, file_name)
                                field_data += [self._load_field_scalar(file_field)]
                            field_data = np.array(field_data)
                            field_name = field_name[:-1]

                        if str(field_snap) not in fields.keys():
                            fields.__dict__[str(field_snap)] = fargopy.Dictobj()
                        size += field_data.nbytes
                        (fields.__dict__[str(field_snap)]).__dict__[f"{field_name}"] = (
                            fargopy.Field(
                                data=field_data,
                                coordinates=self.vars.COORDINATES,
                                domains=self.domains,
                                type=type,
                            )
                        )

                    else:
                        # Store a specific snapshot
                        if int(comps[1]) == snapshot:
                            field_name = comps[0]

                            if type == "scalar":
                                field_data = self._load_field_scalar(file_field)
                            elif type == "vector":
                                field_data = []
                                variables = ["x", "y"]
                                if self.vars.DIM == 3:
                                    variables += ["z"]
                                for i, variable in enumerate(variables):
                                    file_name = (
                                        f"{fluid}{variable}{str(field_snap)}.dat"
                                    )
                                    file_field = os.path.join(
                                        self.output_dir, file_name
                                    )
                                    field_data += [self._load_field_scalar(file_field)]
                                field_data = np.array(field_data)
                                field_name = field_name[:-1]

                            size += field_data.nbytes
                            fields.__dict__[f"{field_name}"] = fargopy.Field(
                                data=field_data,
                                coordinates=self.vars.COORDINATES,
                                domains=self.domains,
                                type=type,
                            )

        else:
            raise ValueError(
                f"No field found with pattern '{pattern}'. Change the fluid"
            )

        if qall:
            fields.snapshots = sorted([int(s) for s in fields.keys() if s != "size"])
        fields.size = size / 1024**2
        return fields
