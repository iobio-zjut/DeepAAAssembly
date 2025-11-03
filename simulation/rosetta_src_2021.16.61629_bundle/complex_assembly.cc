// App headers
#include <devel/init.hh>

// Project Headers
#include <protocols/moves/Mover.hh>
#include <core/conformation/membrane/Span.hh>
#include <core/conformation/membrane/SpanningTopology.hh>
#include <protocols/membrane/util.hh>

#include <core/conformation/Residue.hh>
#include <core/chemical/ResidueType.hh>
#include <core/chemical/ResidueTypeSet.hh>
#include <core/chemical/ChemicalManager.hh>
#include <core/pack/task/TaskFactory.hh>
#include <core/pack/pack_rotamers.hh>

#include <basic/options/option.hh>
#include <basic/options/keys/mp.OptionKeys.gen.hh>
#include <basic/options/keys/in.OptionKeys.gen.hh>
#include <basic/options/keys/out.OptionKeys.gen.hh>
#include <basic/options/keys/relax.OptionKeys.gen.hh>

#include <core/sequence/Sequence.hh>
#include <core/sequence/util.hh>
#include <core/pose/annotated_sequence.hh>

#include <protocols/simple_moves/SwitchResidueTypeSetMover.hh>
#include <protocols/simple_moves/BackboneMover.hh>
#include <protocols/simple_moves/SuperimposeMover.hh>

#include <core/kinematics/MoveMap.hh>
#include <core/kinematics/FoldTree.hh>

#include <core/scoring/ScoreFunctionFactory.hh>
#include <core/scoring/ScoreFunction.hh>
#include <core/scoring/Energies.hh>
#include <core/scoring/constraints/Constraint.hh>
#include <core/scoring/constraints/AtomPairConstraint.hh>
#include <core/scoring/constraints/DihedralConstraint.hh>
#include <core/scoring/constraints/AngleConstraint.hh>
#include <core/scoring/func/SplineFunc.hh>

#include <core/optimization/MinimizerOptions.hh>
#include <core/optimization/AtomTreeMinimizer.hh>

#include <protocols/moves/MonteCarlo.fwd.hh>
#include <protocols/moves/MonteCarlo.hh>
#include <protocols/simple_moves/FragmentMover.hh>
#include <core/fragment/FragSet.hh>
#include <core/fragment/FragmentIO.hh>

#include <protocols/trRosetta/trRosettaOutputs_v1.hh>
#include <protocols/trRosetta/trRosettaProtocol_v1.hh>
#include <protocols/trRosetta_protocols/movers/trRosettaProtocolMover.hh>
#include <protocols/trRosetta_protocols/movers/trRosettaProtocolMoverCreator.hh>

#include <core/chemical/AA.hh>
#include <core/select/residue_selector/ResidueIndexSelector.hh>
#include <core/scoring/RamaPrePro.hh>
#include <core/simple_metrics/metrics/RMSDMetric.hh>
#include <core/simple_metrics/metrics/TotalEnergyMetric.hh>
#include <core/simple_metrics/metrics/TimingProfileMetric.hh>
#include <core/util/SwitchResidueTypeSet.hh>

#include <protocols/trRosetta_protocols/constraint_generators/trRosettaConstraintGenerator.hh>
#include <protocols/simple_moves/MutateResidue.hh>
#include <protocols/residue_selectors/StoredResidueSubsetSelector.hh>
#include <protocols/residue_selectors/StoreResidueSubsetMover.hh>
#include <protocols/simple_moves/bin_transitions/InitializeByBins.hh>
#include <protocols/minimization_packing/MinMover.hh>
#include <protocols/moves/RepeatMover.hh>
#include <protocols/relax/FastRelax.hh>

// Package Headers
#include <apps/benchmark/performance/init_util.hh>
#include <core/types.hh>
#include <numeric/xyzVector.hh>
#include <numeric/constants.hh>
#include <core/id/AtomID.hh>
#include <core/pose/Pose.hh>
#include <core/pose/PDBInfo.hh>
#include <core/import_pose/import_pose.hh>
#include <core/pose/util.hh>
#include <core/pose/variant_util.hh>
#include <core/pose/subpose_manipulation_util.hh>
#include <core/conformation/util.hh>
#include <protocols/jd2/JobDistributor.hh>
#include <protocols/jd2/Job.hh>
#include <basic/Tracer.hh>

// utility headers
#include <utility/excn/Exceptions.hh>
#include <utility/excn/Exceptions.hh>
#include <utility/string_util.hh>
#include <utility/io/ozstream.hh>
#include <utility/string_util.hh>
#include <utility/minmax.hh>
#include <numeric/random/random.hh>
#include <protocols/jd2/util.hh>
#include <core/scoring/rms_util.hh>
#include <core/scoring/rms_util.tmpl.hh>
#include <basic/options/keys/score.OptionKeys.gen.hh>
#include <basic/options/keys/trRosetta.OptionKeys.gen.hh>
#include <basic/tensorflow_manager/util.hh>
#include <utility/tag/Tag.hh>
#include <utility/pointer/memory.hh>

// C++ Headers
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <map>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "pareto.hh"


using core::Real;
using namespace std;
using namespace core;
using namespace core::pose;
using namespace core::conformation;
using namespace basic::options;
using namespace basic::options::OptionKeys;
using namespace protocols::simple_moves;
using namespace core::chemical;
using namespace protocols::moves;
using namespace core::scoring;
using namespace protocols::trRosetta;
using namespace protocols::trRosetta_protocols::movers;
using namespace core::scoring::constraints;
using namespace core::scoring::func;

static basic::Tracer TR( "apps.public.complex_assembly" );

class DomainAssembly : public Mover {

public:

	//////////////////////////
	/// @note Constructors ///
	//////////////////////////

	/// @brief Construct a Default Mover
	DomainAssembly();

	/// @brief Copy Constructor
	/// @details Make a deep copy of this mover object
	DomainAssembly( DomainAssembly const & src );

	/// @brief Assignment Operator
	/// @details Make a deep copy of this mover object, overriding the assignment operator
	DomainAssembly &
	operator=( DomainAssembly const & src );

	/// @brief Destructor
	~DomainAssembly() override;

	///////////////////////////
	/// @note Mover Methods ///
	///////////////////////////

	/// @brief Get the name of this mover
	std::string get_name() const override;

	/// @brief Get protein interface statistics
	void apply( Pose & pose ) override;


private: // parameter

	/// @brief Maximum number of generations (angle)
	Size G = 300;

	/// @brief Population size (pose)
	Size NP1 = 100;

	/// @brief Population size (angle)
	Size NP2 = 100;

	/// @brief Maximum disturbance of initial angle
	Real max_disturbance = 5;

	/// @brief Scaling factor
	Real F = 0.5;

	/// @brief Crossover factor
	Real CR = 0.5;

	/// @brief Number of candidate individual
	Size N_candidate = 100;

	/// @brief Number of output model
	Size N_output = 10;

	/// @brief Lenth of linker
	Size L_linker = 8;

	/// @brief Will this generate distance constraints?
	bool set_generate_dist_constraints_ = true;

	/// @brief Probability cutoff for distance constraints.  Default 0.05.
	core::Real dist_prob_cutoff_ = 0.05;

	/// @brief Probability cutoff for distance constraints in relax.  Default 0.15.
	core::Real ref_dist_prob_cutoff_ = 0.15;

	/// @brief Constraint weight for distance constraints.  Default 1.0.
	core::Real distance_constraint_weight_ = 1.0;

	/// @brief Number of dist bins
	Size const n_dist_bins_ = 37;

	/// @brief Does relax begin?
	bool relax_ = true;


private: // route

	/// @brief Path of distance constraint file
    std::string dist_path1_ = "./distprob.txt";
	std::string dist_path2_ = "./AF3_inter.txt";
	std::string dist_path3_ = "./model1.txt";
	std::string dist_path4_ = "./model2.txt";
	std::string dist_path5_ = "./model3.txt";
	std::string dist_path6_ = "./model4.txt";


    /// @brief Path of output file
	std::string output_path_ = "./trans_mat_5model_bestscore/trans_mat";


private: // data

    /// @brief Length of the full-length protein
	Size nres;

	utility::vector1<double> epitope_prob;
	/// @brief Sequence of the full-length protein: given as fasta
	std::string full_seq_;

	/// @brief Native pose
//	Pose native;

	/// @brief Number of the pose in the -in:file:l list that is the TM domain
	Size tmpdb_{};

	/// @brief Vector of pose input files
	utility::vector1< std::string > infiles_;

	/// @brief Vector of poses_ that are read in from PDB files
	utility::vector1< core::pose::Pose > poses_;

	utility::vector1< core::pose::Pose > chains_;

	/// @brief Vector of sequences_ in poses_
	utility::vector1< std::string > sequences_;

	/// @brief Vector of offsets_, i.e. residue numbers along the full-length sequence
	///   for which the PDB poses_ start
	utility::vector1< Size > offsets_;

	/// @brief Vector of linker sequences_ to model
	utility::vector1< std::string > linkers_;

	/// @brief Vector of index of start and end of movable residues
	utility::vector1< std::pair< Size, Size > > movable_residues_;

	/// @brief Vector of index of start and end of immovable residues
	utility::vector1< std::pair< Size, Size > > immovable_residues_;

	/// @brief Movemap of which residues can move (i.e. are in linkers_ +/-1 residue)
	core::kinematics::MoveMap movemap_;

	/// @brief score function
	core::scoring::ScoreFunctionOP sfxn_;

	/// @brief full-atom score function
	core::scoring::ScoreFunctionOP sfxn_fullatom_;

	/// @brief The dist constraints from the neural network
	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constraints;
	
	utility::vector1< utility::vector1< utility::vector1< utility::vector1< utility::vector1< Real > > > > > dist_probs;

	/// @brief Affine transformations from the neural network
	utility::vector1< utility::vector1< Real > > affine_transformations;

	/// @brief save score of pose
	struct SCORE
	{
		Real total_score;
		Real rmsd;

		SCORE() { total_score = 0.0; rmsd = 0.0; }
		SCORE( Real v1, Real v2 ) : total_score(v1), rmsd(v2) {}
	};

	struct ScoreIndex {
    Real score;
    size_t index;
    };

	/// @brief population of full pose
	utility::vector1< Pose > population_pose_;

	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_1;
	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_2;
	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_3;
	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_4;
	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_5;
	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_6;

	/// @brief population of pose score
	utility::vector1< Real > population_score_1;
	utility::vector1< Real > population_score_2;
	utility::vector1< Real > population_score_3;
	utility::vector1< Real > population_score_4;
	utility::vector1< Real > population_score_5;
	utility::vector1< Real > population_score_6;

	/// @brief rotation axis
	utility::vector1< std::pair< numeric::xyzVector< Real >, numeric::xyzVector< Real > > > rotation_axis_;

	/// @brief rotation point
	utility::vector1< numeric::xyzVector< Real > > rotation_point_;

	/// @brief sort by total score
	multimap< Real, Pose > sort_total_score_pose_;

	/// @brief relaxed model
	multimap< Real, Pose > sort_ref_total_score_pose_;


	utility::vector1< utility::vector1< utility::vector1< Real > > > lframe_;

	utility::vector1< utility::vector1< utility::vector1< Real > > > lcoord_CA_;


	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constr2_;
    utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constr3_;
	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constr4_;
	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constr5_;
	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constr6_;


private: // methods

	/// @brief Register Options with JD2
	void register_options();

	/// @brief Initialize Mover options from the comandline
	void init_from_cmd();

    ///@brief pareto
    utility::vector1<utility::vector1<utility::vector1<Real>>> delete_unwanted_rot_tra(utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Size> &target_index);

    void Pareto_method1(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
    void Pareto_method2(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
	void Pareto_method3(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
	void Pareto_method4(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
	void Pareto_method5(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
	void Pareto_method6(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);

	/// @brief create residue
	ResidueOP create_residue_from_olc( char olc );

	/// @brief create residue
	Residue create_residue_from_resn( Pose & pose, Size resnumber );

	/// @brief set random torsion
	void set_random_torsion( Pose & pose, Size resn );

	utility::vector1< utility::vector1< Real > > set_random_rot_tra();

	/// @brief Print score to cout
	void print_score( Pose & pose, core::scoring::ScoreFunctionOP sfxn );

	/// @brief calculate RMSD of pose
	Real cal_rmsd( Pose & pose );

	/// @brief calculate total score of pose
	Real cal_total_score( Pose & pose );

	/// @brief calculate ref total score of pose
	Real cal_ref_total_score( Pose & pose );

	/// @brief calculate distance score of pose
//	Real cal_dscore( Pose & pose );

    /// @brief defines reference frame per residue using backbone atoms
    utility::vector1< utility::vector1< Real > > set_lframe( Pose & pose, Size res );

    /// @brief calculate frame aligned point error (CA)
    Real frame_aligned_point_error_CA( Pose & pose );

    /// @brief calculate frame aligned point error (bb)
    Real frame_aligned_point_error_bb( Pose & pose );

    Real score_rot_tra( utility::vector1< utility::vector1< Real > > rot_tra_ );

    Real score_rot_tra_v1( utility::vector1< utility::vector1< Real > > rot_tra_ );

    Real score_rot_tra_v2( utility::vector1< utility::vector1< Real > > rot_tra_ );
    Real score_rot_tra_v3( utility::vector1< utility::vector1< Real > > rot_tra_ );
    Real score_rot_tra_v4( utility::vector1< utility::vector1< Real > > rot_tra_ );
    Real score_rot_tra_v5( utility::vector1< utility::vector1< Real > > rot_tra_ );
    Real score_rot_tra_v6( utility::vector1< utility::vector1< Real > > rot_tra_ );

    utility::vector1< utility::vector1< Real > > rot_tra_to_affine( utility::vector1< Real > rot_tra );

    utility::vector1< utility::vector1< Real > > EulerAngles_to_rotationMatrix( Real euler_x, Real euler_y, Real euler_z );

    utility::vector1< Real > spherical_to_translationVector( Real spher_r, Real spher_t, Real spher_p );

    utility::vector1< utility::vector1< Real > > matrix_multiply( utility::vector1< utility::vector1< Real > > arrA, utility::vector1< utility::vector1< Real > > arrB );

	/// @brief Transform the index of residue number (full to part)
	Size index_f2p( Size f_index );

	/// @brief Transform the index of residue number (part to full)
	Size index_p2f( bool is_axis, Size p_index );

	/// @brief Accepted with Boltzmann probability
	bool boltzmann_accept( const Real targetEnergy, const Real trialEnergy, Real recipocal_KT );

	bool greedy_accept( const Real targetEnergy, const Real trialEnergy);

	/// @brief Score of rotation angle
//	Real score_rotation_angle( utility::vector1< Real > & rotation_angle );

	/// @brief Score of rotation angle
	Real score_rotation_angle_v1( Pose & pose, utility::vector1< Real > & rotation_angle );

	/// @brief Solving the rotation angle by differential evolution algorithm
	utility::vector1< Real > rotation_angle_optimization( Size index );

	/// @brief Given a pose, generate the constraints
	core::scoring::constraints::ConstraintCOPs generate_constraints( Pose const & pose ) const;
	
	/// @brief Generate the distance constraints
	void generate_dist_constraints(
		utility::vector1 <core::scoring::constraints::ConstraintCOP> & outputvec,
		utility::vector1< Real > const & dist_bins_background,
		utility::vector1< Real > const & dist_bins_vect,
		core::id::AtomID const & cb_atom_i,
		core::id::AtomID const & cb_atom_j,
		Size const n_dist_bins,
		Size const n_dist_bins_model,
		Size const ires,
		Size const jres,
		Real const prob_cutoff
	) const;

	/// @brief Remove previously-added constraints from the pose
	void remove_constraints(
		utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
		Pose & pose
	) const;

	/// @brief Add constraints from a list to the pose, if the constraints are between residues that are
	/// separated by at least min_seqsep but less than max_seqsep residues.
	/// @details This does not clear constraints from the pose.
	/// @note It is "less than" max_seqsep, not "less than or equal to".
	void add_constraints_to_pose(
		Pose & pose,
		utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
		Size const min_seqsep,
		Size const max_seqsep,
		bool const skip_glycine_positions = false
	) const;

	/// @brief Given a constraint, determine if it is an AtomPairConstraint, an
	/// AngleConstraint, or a DihedralConstraint, pull out the pair of residues
	/// that are constrained, and return the pair.
	/// @details Throws if type is unrecognized or if more than two residues are
	/// constrained.  Values of res1 and res2 are overwritten by this operation.
	void get_residues_from_constraint(
		Size & res1,
		Size & res2,
		core::scoring::constraints::ConstraintCOP const & cst
	) const;

	/// @brief Given a constraint, determine if it is an AtomPairConstraint, pull
	/// out the pair of residues that are constrained, and return the pair.
	/// @details If successful, values of res1 and res2 are overwritten by this
	/// operation, and the function returns "true".  Otherwise, values are not
	/// altered, and the function returns "false".
	bool get_residues_from_atom_pair_constraint(
		Size & res1,
		Size & res2,
		core::scoring::constraints::ConstraintCOP const & cst
	) const;

	/// @brief Do all-atom FastRelax refinement.
	void do_fullatom_refinement(
		core::pose::Pose & pose,
		utility::vector1< bool > move_bb
	) const;

};

////////////////////////////////////////////////////////////////////////////////

//////////////////////////
/// @note Constructors ///
//////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Construct a Default Position Mover
DomainAssembly::DomainAssembly() :
	Mover(),
	full_seq_(),
	infiles_(),
	poses_(),
	sequences_(),
	offsets_(),
	linkers_(),
	movemap_()
{}

////////////////////////////////////////////////////////////////////////////////
/// @brief Copy Constructor
/// @details Make a deep copy of this mover object
DomainAssembly::DomainAssembly( DomainAssembly const & src ) :
	Mover( src ),
	full_seq_( src.full_seq_ ),
	tmpdb_( src.tmpdb_ ),
	infiles_( src.infiles_ ),
	poses_( src.poses_ ),
	sequences_( src.sequences_ ),
	offsets_( src.offsets_ ),
	linkers_( src.linkers_ ),
	movemap_( src.movemap_ )
{}

////////////////////////////////////////////////////////////////////////////////
/// @brief Assignment Operator
/// @details Make a deep copy of this mover object, overriding the assignment operator
DomainAssembly &
DomainAssembly::operator=( DomainAssembly const & src )
{
	// Abort self-assignment.
	if ( this == &src ) {
		return *this;
	}

	// Otherwise, create a new object
	return *( new DomainAssembly( *this ) );
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Destructor
DomainAssembly::~DomainAssembly() = default;

///////////////////////////
/// @note Mover Methods ///
///////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Get the name of this mover
std::string
DomainAssembly::get_name() const {
	return "DomainAssembly";
}

////////////////////////////////////////////////////////////////////////////////
/// @brief run domain assembly
void DomainAssembly::apply( Pose & pose ) {

	TR << "Calling DomainAssembly" << std::endl;

	register_options();
	init_from_cmd();

	// clear vector contents, needed for running things in parallel
	pose.clear();
	poses_.clear();
	sequences_.clear();
	offsets_.clear();


	////////////////////////////////////////////////////////////////////////////////////////
	/// @note SETUP

	// read in poses_
	utility::vector1< core::pose::Pose > poses_ = core::import_pose::poses_from_files( infiles_ );

	chains_ = poses_;

	// get sequences_ from chains_
	for ( Size i = 1; i <= chains_.size(); ++i ) {
		sequences_.push_back( chains_[i].sequence() );
	}

	TR << "CHAINS " << chains_.size() << std::endl;
	TR << "SEQUENCES " << sequences_.size() << std::endl;

	for ( Size i = 1; i <= sequences_.size(); ++i ) {
		std::string this_seq = sequences_[i];
		TR << "sequence " << i <<": " << sequences_[i] << std::endl;
	}



////////////////////////////////////////////////////////////////////////////////////////
	/// @note READ CONSTRAINTS——1

	ifstream dist_path1( dist_path1_.c_str() );
	if ( !dist_path1 ) {
		TR << "Please provide distance constraint2 file!" << std::endl;
		exit(0);
	}

    // load distance constraints
    TR << "loading distance constraint2..." << std::endl;
//	utility::vector1< utility::vector1< utility::vector1< utility::vector1< utility::vector1< Real > > > > > dist_probs;
	dist_probs.resize(sequences_[1].size(), utility::vector1< utility::vector1< utility::vector1< utility::vector1< Real > > > >( sequences_[2].size(), utility::vector1< utility::vector1< utility::vector1< Real > > >( 1, utility::vector1< utility::vector1< Real > >(2))));

	std::string dist_line0;
	while ( getline( dist_path1, dist_line0 ) ) {
		istringstream data( dist_line0 );

		Size r1, r2, count;
		data >> r1 >> r2 >> count;

		dist_probs[r1][r2][1][1].resize(count);
    	dist_probs[r1][r2][1][2].resize(count);

		Real dist_value;
		Real weight;

		for (Size n = 1; n <= count; ++n) {

			data >> dist_value;  // 读取浮点数
			dist_probs[r1][r2][1][1][n] = dist_value;
		}

		// 读取权重
		for (Size n = 1; n <= count; ++n) {

			data >> weight;  // 读取权重值
			dist_probs[r1][r2][1][2][n] = weight;
		}


	dist_path1.close();


////////////////////////////////////////////////////////////////////////////////////////
/// @note READ CONSTRAINTS——(2~6)

    for (int i = 2; i <= 6; ++i) {
        std::string dist_path_str;
        utility::vector1< utility::vector1< utility::vector1< Real > > >* dist_constr_ptr = nullptr;

        // 选择对应文件路径与目标容器
        switch (i) {
            case 2:
                dist_path_str = dist_path2_;
                dist_constr_ptr = &dist_constr2_;
                break;
            case 3:
                dist_path_str = dist_path3_;
                dist_constr_ptr = &dist_constr3_;
                break;
            case 4:
                dist_path_str = dist_path4_;
                dist_constr_ptr = &dist_constr4_;
                break;
            case 5:
                dist_path_str = dist_path5_;
                dist_constr_ptr = &dist_constr5_;
                break;
            case 6:
                dist_path_str = dist_path6_;
                dist_constr_ptr = &dist_constr6_;
                break;
            default:
                continue;
        }

        ifstream dist_file(dist_path_str.c_str());
        if (!dist_file) {
            TR << "Please provide distance constraint file for " << i << "!" << std::endl;
            exit(0);
        }

        TR << "loading distance constraint" << i << "..." << std::endl;

        std::string dist_line;
        utility::vector1< utility::vector1< Real > > dist_constr_tmp;
        Size line_number = 1;
        Size chain_number = 2;

        while (getline(dist_file, dist_line)) {
            istringstream data(dist_line);
            utility::vector1<Real> line_dist_constr;
            Real distance_value;

            for (Size m = 1; m <= sequences_[chain_number].size(); ++m) {
                data >> distance_value;
                line_dist_constr.push_back(distance_value);
            }
            dist_constr_tmp.push_back(line_dist_constr);

            if (line_number == sequences_[1].size()) {
                line_number = 1;
                chain_number++;
                dist_constr_ptr->push_back(dist_constr_tmp);
                dist_constr_tmp.clear();
            } else {
                line_number++;
            }
        }

        dist_file.close();
    }


	////////////////////////////////////////////////////////////////////////////////////////
	/// @note POPULATION INITIALIZATION

    TR << "population initialization..." << std::endl;

	for ( Size n = 1; n <= NP1; ++n ) {


        // 创建指针数组，指向各个种群的变量
        utility::vector1< utility::vector1< Real > >* population_rot_tra_arr[6] = { &population_rot_tra_1, &population_rot_tra_2, &population_rot_tra_3, &population_rot_tra_4, &population_rot_tra_5, &population_rot_tra_6 };
        utility::vector1< Real >* population_score_arr[6] = { &population_score_1, &population_score_2, &population_score_3, &population_score_4, &population_score_5, &population_score_6 };

        // 创建对应的评分函数指针数组
        Real (*score_func_arr[6])( const utility::vector1< utility::vector1< Real > >& ) = {
            score_rot_tra_v1, score_rot_tra_v2, score_rot_tra_v3, score_rot_tra_v4, score_rot_tra_v5, score_rot_tra_v6
        };

        // 循环初始化六个种群
        for ( int i = 0; i < 6; ++i ) {
            utility::vector1< utility::vector1< Real > > random_rot_tra( set_random_rot_tra() );
            Real score_random_rot_tra = score_func_arr[i]( random_rot_tra );

            population_rot_tra_arr[i]->push_back( random_rot_tra );
            population_score_arr[i]->push_back( score_random_rot_tra );
        }

	}


	////////////////////////////////////////////////////////////////////////////////////////
	/// @note POPULATION OPTIMIZATION

    // 创建指针数组，指向各个种群的变量
    // 标量分数
    Real bestScore1, bestScore2, bestScore3, bestScore4, bestScore5, bestScore6;
    // 索引向量
    utility::vector1<Size> index_vec1, index_vec2, index_vec3, index_vec4, index_vec5, index_vec6;
    // 最佳个体
    utility::vector1< utility::vector1< Real > > bestIndiv1, bestIndiv2, bestIndiv3, bestIndiv4, bestIndiv5, bestIndiv6;
    // 临时旋转平移
    utility::vector1< utility::vector1< utility::vector1< Real > > > tmp_rot_tra1, tmp_rot_tra2, tmp_rot_tra3, tmp_rot_tra4, tmp_rot_tra5, tmp_rot_tra6;
    // 当前旋转平移
    utility::vector1< utility::vector1< utility::vector1< Real > > > my_rot_tra1, my_rot_tra2, my_rot_tra3, my_rot_tra4, my_rot_tra5, my_rot_tra6;

    Real* bestScore_arr[6] = { &bestScore1, &bestScore2, &bestScore3, &bestScore4, &bestScore5, &bestScore6 };
    utility::vector1<Size>* index_vec_arr[6] = { &index_vec1, &index_vec2, &index_vec3, &index_vec4, &index_vec5, &index_vec6 };
    utility::vector1< utility::vector1< Real > >* bestIndiv_arr[6] = { &bestIndiv1, &bestIndiv2, &bestIndiv3, &bestIndiv4, &bestIndiv5, &bestIndiv6 };
    utility::vector1< utility::vector1< utility::vector1< Real > > >* tmp_rot_tra_arr[6] = { &tmp_rot_tra1, &tmp_rot_tra2, &tmp_rot_tra3, &tmp_rot_tra4, &tmp_rot_tra5, &tmp_rot_tra6 };
    utility::vector1< utility::vector1< utility::vector1< Real > > >* my_rot_tra_arr[6] = { &my_rot_tra1, &my_rot_tra2, &my_rot_tra3, &my_rot_tra4, &my_rot_tra5, &my_rot_tra6 };

    // 循环初始化
    for ( int i = 0; i < 6; ++i ) {
        *bestScore_arr[i] = 1000000.0;
        index_vec_arr[i]->clear();
        bestIndiv_arr[i]->clear();
        tmp_rot_tra_arr[i]->clear();
        my_rot_tra_arr[i]->clear();
    }


//    Real KT = 100.0 ;
    for ( Size g = 1; g <= G; ++g ) {
		for ( Size n = 1; n <= NP1; ++n ) {

			Size base( numeric::random::rg().random_range( 1, NP1 ) );
			Size rand1( numeric::random::rg().random_range( 1, NP1 ) );
			Size rand2( numeric::random::rg().random_range( 1, NP1 ) );

			while (rand1 == base)
				rand1 = numeric::random::rg().random_range( 1, NP1 );
			while (rand2 == base || rand2 == rand1)
				rand2 = numeric::random::rg().random_range( 1, NP1 );

            // 将六个种群的相关变量统一放入vector中
            std::vector< utility::vector1< utility::vector1< Real > >* > population_rot_tra_vec = {
                &population_rot_tra_1, &population_rot_tra_2, &population_rot_tra_3,
                &population_rot_tra_4, &population_rot_tra_5, &population_rot_tra_6
            };

            std::vector< utility::vector1< Real >* > population_score_vec = {
                &population_score_1, &population_score_2, &population_score_3,
                &population_score_4, &population_score_5, &population_score_6
            };

            std::vector< utility::vector1< utility::vector1< utility::vector1< Real > > >* > tmp_rot_tra_vec = {
                &tmp_rot_tra1, &tmp_rot_tra2, &tmp_rot_tra3,
                &tmp_rot_tra4, &tmp_rot_tra5, &tmp_rot_tra6
            };

            std::vector< Real* > bestScore_vec = {
                &bestScore1, &bestScore2, &bestScore3,
                &bestScore4, &bestScore5, &bestScore6
            };

            std::vector< utility::vector1< utility::vector1< Real > >* > bestIndiv_vec = {
                &bestIndiv1, &bestIndiv2, &bestIndiv3,
                &bestIndiv4, &bestIndiv5, &bestIndiv6
            };

            // 循环处理六个种群
            for ( int pop_id = 0; pop_id < 6; ++pop_id ) {

                auto &population_rot_tra = *population_rot_tra_vec[pop_id];
                auto &population_score = *population_score_vec[pop_id];
                auto &tmp_rot_tra_ref = *tmp_rot_tra_vec[pop_id];
                auto &bestScore_ref = *bestScore_vec[pop_id];
                auto &bestIndiv_ref = *bestIndiv_vec[pop_id];

                // -------------------- Mutation --------------------
                utility::vector1< utility::vector1< Real > > mutant_rot_tra_;
                for ( Size i = 1; i < tmpdb_; ++i ) {
                    utility::vector1< Real > mutant_rot_tra;
                    for ( Size d = 1; d <= 6; ++d ) {
                        mutant_rot_tra.push_back(
                            population_rot_tra[base][i][d] +
                            F * (population_rot_tra[rand1][i][d] - population_rot_tra[rand2][i][d])
                        );
                    }
                    if ( cos(mutant_rot_tra[2]) < 0 )
                        mutant_rot_tra[2] = numeric::random::uniform() * 180 - 90;

                    mutant_rot_tra_.push_back(mutant_rot_tra);
                }

                // -------------------- Crossover --------------------
                utility::vector1< utility::vector1< Real > > cross_rot_tra_;
                for ( Size i = 1; i < tmpdb_; ++i ) {
                    utility::vector1< Real > cross_rot_tra;
                    Size rand_3 = numeric::random::rg().random_range(1,6);
                    for ( Size d = 1; d <= 6; ++d ) {
                        if ( numeric::random::rg().uniform() <= CR || d == rand_3 )
                            cross_rot_tra.push_back(mutant_rot_tra_[i][d]);
                        else
                            cross_rot_tra.push_back(population_rot_tra[n][i][d]);
                    }
                    cross_rot_tra_.push_back(cross_rot_tra);
                }

                // -------------------- Update --------------------
                Real targetScore = population_score[n];

                // 使用函数指针数组统一调用score函数
                typedef Real (*ScoreFunc)( const utility::vector1< utility::vector1< Real > > & );
                ScoreFunc score_funcs[6] = { score_rot_tra_v1, score_rot_tra_v2, score_rot_tra_v3,
                                             score_rot_tra_v4, score_rot_tra_v5, score_rot_tra_v6 };

                Real trialScore = score_funcs[pop_id](cross_rot_tra_);

                bool success = boltzmann_accept(targetScore, trialScore, 1.0);
                if (success) {
                    population_rot_tra[n] = cross_rot_tra_;
                    tmp_rot_tra_ref.push_back(cross_rot_tra_);
                    population_score[n] = trialScore;
                } else {
                    tmp_rot_tra_ref.push_back(population_rot_tra[n]);
                }

                if (population_score[n] <= bestScore_ref) {
                    bestScore_ref = population_score[n];
                    bestIndiv_ref = population_rot_tra[n];
                }
            }


		}

        // 创建Pareto函数指针数组
        void (*pareto_func_arr[6])(utility::vector1<Size>&,
                                   utility::vector1< utility::vector1< utility::vector1<Real> > >&,
                                   utility::vector1<Real>&) = {
            Pareto_method1, Pareto_method2, Pareto_method3, Pareto_method4, Pareto_method5, Pareto_method6
        };

        // 创建各种群相关的变量指针数组
        utility::vector1< utility::vector1< utility::vector1<Real> > >* tmp_rot_tra_arr[6] = {
            &tmp_rot_tra1, &tmp_rot_tra2, &tmp_rot_tra3, &tmp_rot_tra4, &tmp_rot_tra5, &tmp_rot_tra6
        };

        utility::vector1< utility::vector1< utility::vector1<Real> > >* my_rot_tra_arr[6] = {
            &my_rot_tra1, &my_rot_tra2, &my_rot_tra3, &my_rot_tra4, &my_rot_tra5, &my_rot_tra6
        };

        utility::vector1<Real>* population_score_arr[6] = {
            &population_score_1, &population_score_2, &population_score_3, &population_score_4, &population_score_5, &population_score_6
        };

        utility::vector1< utility::vector1<Real> >* population_rot_tra_arr[6] = {
            &population_rot_tra_1, &population_rot_tra_2, &population_rot_tra_3, &population_rot_tra_4, &population_rot_tra_5, &population_rot_tra_6
        };

        utility::vector1<Size>* index_vec_arr[6] = {
            &index_vec1, &index_vec2, &index_vec3, &index_vec4, &index_vec5, &index_vec6
        };

        // bestScore数组
        Real* bestScore_arr[6] = { &bestScore1, &bestScore2, &bestScore3, &bestScore4, &bestScore5, &bestScore6 };

        // 循环处理六个种群的Pareto与Exchange
        for ( int i = 0; i < 6; ++i ) {

            TR << "种群" << (i+1) << "[ " << g << " / " << G << " ]...score: " << *bestScore_arr[i] << std::endl;

            // 调用对应的Pareto方法
            pareto_func_arr[i](*index_vec_arr[i], *tmp_rot_tra_arr[i], *population_score_arr[i]);
            TR << "种群" << (i+1) << " PARETO get " << index_vec_arr[i]->size() << " rot_tra" << std::endl;

            // 删除不需要的rot_tra
            *my_rot_tra_arr[i] = delete_unwanted_rot_tra(*tmp_rot_tra_arr[i], *index_vec_arr[i]);
            utility::vector1< utility::vector1< utility::vector1<Real> > >().swap(*tmp_rot_tra_arr[i]);

            // 构建ScoreIndex数组并排序
            std::vector<ScoreIndex> score_indices;
            for ( size_t j = 1; j <= population_score_arr[i]->size(); ++j ) {
                score_indices.push_back({ (*population_score_arr[i])[j], j });
            }
            std::sort(score_indices.begin(), score_indices.end(), [](const ScoreIndex& a, const ScoreIndex& b) {
                return a.score < b.score;
            });

            // 更新population_rot_tra
            for ( size_t j = 1; j <= my_rot_tra_arr[i]->size(); ++j ) {
                size_t original_index = score_indices[population_score_arr[i]->size() - j].index;
                (*population_rot_tra_arr[i])[original_index] = (*my_rot_tra_arr[i])[j];
            }
        }


	}


	////////////////////////////////////////////////////////////////////////////////////////
	/// @note OUTPUT ROTATION AND TRANSLATION
	TR << "output rotation and translation..." << std::endl;
	for ( Size i = 1; i < tmpdb_; ++i ) {
        // 创建 bestScore 数组
        Real* bestScore_arr[6] = { &bestScore1, &bestScore2, &bestScore3, &bestScore4, &bestScore5, &bestScore6 };

        // 创建 bestIndiv 数组
        utility::vector1< utility::vector1<Real> >* bestIndiv_arr[6] = {
            &bestIndiv1, &bestIndiv2, &bestIndiv3, &bestIndiv4, &bestIndiv5, &bestIndiv6
        };

        // 循环输出六个种群的 best
        for (int pop = 0; pop < 6; ++pop) {
            std::string best_output_path = output_path_ + "_best_pop" + std::to_string(pop + 1) + "_" + std::to_string(*bestScore_arr[pop]) + ".txt";
            ofstream output_file(best_output_path.c_str());

            // 构建旋转矩阵
            utility::vector1< utility::vector1<Real> > rotMat_best(
                EulerAngles_to_rotationMatrix(
                    (*bestIndiv_arr[pop])[i][1],
                    (*bestIndiv_arr[pop])[i][2],
                    (*bestIndiv_arr[pop])[i][3]
                )
            );
            // 构建平移向量
            utility::vector1<Real> traVec_best{
                (*bestIndiv_arr[pop])[i][4],
                (*bestIndiv_arr[pop])[i][5],
                (*bestIndiv_arr[pop])[i][6]
            };

            output_file << "rotMat: " << rotMat_best << "    traVec: " << traVec_best << std::endl;
            output_file.close();
        }

	}

    // 创建 my_rot_tra 数组指针
    auto my_rot_tra_arr = std::array<utility::vector1< utility::vector1< utility::vector1<Real> > >*, 6>{
        &my_rot_tra1, &my_rot_tra2, &my_rot_tra3, &my_rot_tra4, &my_rot_tra5, &my_rot_tra6
    };

    // 循环处理六个种群的pareto输出
    for (int pop = 0; pop < 6; ++pop) {
        auto& my_rot = *my_rot_tra_arr[pop];

        for (Size i = 1; i <= my_rot.size(); ++i) {
            for (Size j = 1; j < tmpdb_; ++j) {

                std::string new_output_path = output_path_ + "_" + std::to_string(i) + "_pareto" + std::to_string(pop + 1) + ".txt";
                ofstream output_file(new_output_path.c_str());

                utility::vector1< utility::vector1<Real> > rotMat(
                    EulerAngles_to_rotationMatrix(my_rot[i][j][1], my_rot[i][j][2], my_rot[i][j][3])
                );
                utility::vector1<Real> traVec{
                    my_rot[i][j][4], my_rot[i][j][5], my_rot[i][j][6]
                };

                output_file << "rotMat: " << rotMat << "    traVec: " << traVec << std::endl;
                output_file.close();
            }
        }
    }

} // apply

////////////////////////////////////////////////////////////////////////////////
/// @brief Register Options with JD2
void DomainAssembly::register_options() {

	using namespace basic::options;
	option.add_relevant( OptionKeys::in::file::fasta );
	option.add_relevant( OptionKeys::mp::assembly::poses );
//	option.add_relevant( OptionKeys::mp::assembly::TM_pose_number );
	option.add_relevant( OptionKeys::relax::range::angle_max );
//	option.add_relevant( OptionKeys::in::file::native );

} // register options

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialize Mover options from the comandline
void DomainAssembly::init_from_cmd() {

	using namespace basic::options;

	// cry if PDB list not given
	if ( option[OptionKeys::in::file::fasta].user() ) {
		full_seq_ = core::sequence::read_fasta_file( option[ OptionKeys::in::file::fasta ]()[1] )[1]->sequence();
		TR << "Read in fasta file " << option[OptionKeys::in::file::fasta]()[1] << std::endl;
	} else {
		throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide fasta file with -in:file:fasta!");
	}

	// read in PDB list
	if ( option[mp::assembly::poses].user() ) {
		utility::vector1< std::string > domain_path_ = basic::options::option[mp::assembly::poses]();
		std::string domain_path = domain_path_[1];

		for ( Size n = 1; n < 100 ; ++n ) {
		    stringstream in_pose;
		    in_pose << domain_path << "/dom" << n << ".pdb";
		    string pdb_route;
		    in_pose >> pdb_route;

		    ifstream pdb_route_( pdb_route.c_str() );
	        if ( pdb_route_ )
		        infiles_.push_back( pdb_route );
		}
		if ( infiles_.size() == 0 )
		    throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide domain PDB files!");

	} else {
		throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide route of domain PDB files!");
	}


} // init from commandline

////////////////////////////////////////////////////////////////////////////////
/// @brief create residue
Residue DomainAssembly::create_residue_from_resn( Pose & pose, Size resnumber ) {

	using namespace core::conformation;

	std::string name3 = pose.residue( resnumber ).name3();

	ResidueTypeSetCOP residue_set( ChemicalManager::get_instance()->residue_type_set( core::chemical::FA_STANDARD ) );
	ResidueType const & rsd_type( residue_set->name_map( name3 ) );

	// true: preserve Cbeta
	ResidueOP rsd( ResidueFactory::create_residue( rsd_type, pose.residue( resnumber ), pose.conformation(), true ) );

	Residue rsd1( *rsd );

	return rsd1;

} // create residue from tlc

////////////////////////////////////////////////////////////////////////////////
/// @brief create residue
ResidueOP DomainAssembly::create_residue_from_olc( char olc ) {

	using namespace core::chemical;
	ResidueTypeSetCOP const & residue_set( ChemicalManager::get_instance()->residue_type_set( core::chemical::FA_STANDARD ) );

	ResidueTypeCOP rsd_type( residue_set->get_representative_type_name1( olc ) );
	ResidueType const & res( *rsd_type );
	ResidueOP rsd( ResidueFactory::create_residue( res ) );

	return rsd;
} // create residue from OLC

////////////////////////////////////////////////////////////////////////////////
/// @brief set random torsion
void DomainAssembly::set_random_torsion( Pose & pose, Size resn ) {

	// get random phi and psi
	core::Real random_phi( numeric::random::uniform() * 360 - 180 );
	core::Real random_psi( numeric::random::uniform() * 360 - 180 );

	// set phi and psi to random and omega to 180
	pose.set_phi( resn, random_phi );
	pose.set_psi( resn, random_psi );
	pose.set_omega( resn, 180 );

} // set random torsion

////////////////////////////////////////////////////////////////////////////////
/// @brief set random rotation and translation
utility::vector1< utility::vector1< Real > > DomainAssembly::set_random_rot_tra() {

    utility::vector1< utility::vector1< Real > > random_rot_tra_;
    for ( Size n = 1; n < tmpdb_; ++n ) {

        utility::vector1< Real > random_rot_tra;

    	// get random euler angles
	    core::Real random_euler_x( numeric::random::uniform() * 360 - 180 );
	    core::Real random_euler_y( numeric::random::uniform() * 180 - 90 );
	    core::Real random_euler_z( numeric::random::uniform() * 360 - 180 );

	    // get random spherical radius and angles
	    core::Real random_spherical_radius( numeric::random::uniform() * 200 - 100 );
	    core::Real random_spherical_theta( numeric::random::uniform() * 200 - 100 );
	    core::Real random_spherical_phi( numeric::random::uniform() * 200 - 100 );

	    random_rot_tra.push_back( random_euler_x );
	    random_rot_tra.push_back( random_euler_y );
	    random_rot_tra.push_back( random_euler_z );
	    random_rot_tra.push_back( random_spherical_radius );
	    random_rot_tra.push_back( random_spherical_theta );
	    random_rot_tra.push_back( random_spherical_phi );


	    random_rot_tra_.push_back( random_rot_tra );
    }

    return random_rot_tra_;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Print score to cout
void DomainAssembly::print_score( Pose & pose, core::scoring::ScoreFunctionOP sfxn ) {

	// print energies and iteration
	Real tot_score = ( *sfxn )( pose );
	Real fa_rep = pose.energies().total_energies()[ scoring::fa_rep ];
	Real fa_atr = pose.energies().total_energies()[ scoring::fa_atr ];

	pose.energies().show_total_headers( TR );
	TR << std::endl;
	pose.energies().show_totals( TR );
	TR << std::endl;

	TR << "tot: " << tot_score << " fa_rep: " << fa_rep << " fa_atr: " << fa_atr << std::endl;

} // print score

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate RMSD
Real DomainAssembly::cal_rmsd( Pose & pose ) {
	Real rmsd;
//	rmsd = core::scoring::CA_rmsd( native, pose );
    rmsd = -1.0;
	return rmsd;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate total score
Real DomainAssembly::cal_total_score( Pose & pose ) {
	Real total_score;
	total_score = ( *sfxn_ )( pose );
	return total_score;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate ref total score
Real DomainAssembly::cal_ref_total_score( Pose & pose ) {
	Real ref_total_score;
	ref_total_score = ( *sfxn_fullatom_ )( pose );
	return ref_total_score;
}


////////////////////////////////////////////////////////////////////////////////
/// @brief defines reference frame per residue using backbone atoms
utility::vector1< utility::vector1< Real > > DomainAssembly::set_lframe( Pose & pose, Size res ) {

    utility::vector1< utility::vector1< Real > > lframe;

    numeric::xyzVector< Real > N = pose.residue(res).xyz("N");
    numeric::xyzVector< Real > CA = pose.residue(res).xyz("CA");
    numeric::xyzVector< Real > C = pose.residue(res).xyz("C");

    utility::vector1< Real > v1;
    utility::vector1< Real > v2;
    utility::vector1< Real > e1;
    utility::vector1< Real > u2;
    utility::vector1< Real > e2;
    utility::vector1< Real > e3;

    for ( Size i = 0; i < 3; ++i ) {
        v1.push_back( C[i] - CA[i] );
        v2.push_back( N[i] - CA[i] );
    }

    Real v1_norm = sqrt( v1[1] * v1[1] + v1[2] * v1[2] + v1[3] * v1[3] );

    for ( Size i = 1; i <= 3; ++i )
        e1.push_back( v1[i] / v1_norm );

    Real e1_mul_v2 = e1[1] * v2[1] + e1[2] * v2[2] + e1[3] * v2[3];

    for ( Size i = 1; i <= 3; ++i )
        u2.push_back( v2[i] - e1[i] * e1_mul_v2 );

    Real u2_norm = sqrt( u2[1] * u2[1] + u2[2] * u2[2] + u2[3] * u2[3] );

    for ( Size i = 1; i <= 3; ++i )
        e2.push_back( u2[i] / u2_norm );

    e3.push_back( e1[2] * e2[3] - e1[3] * e2[2] );
    e3.push_back( e1[3] * e2[1] - e1[1] * e2[3] );
    e3.push_back( e1[1] * e2[2] - e1[2] * e2[1] );

    lframe.push_back( e1 );
    lframe.push_back( e2 );
    lframe.push_back( e3 );

    return lframe;
}




Real DomainAssembly::score_rot_tra_v1( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    // Real total_dist_diff2 = 0;
	Real total_dist_diff_temp = 0;
	Real total_dist_diff2_final = 0;

    Real count2 = 0;
    // Real w_total = 0;
//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_probs[res1][res2][1][1].size() > 0 ) {
//                        std::cout << "[res1,res2]: " << res1 <<" "<< res2 << "...dist_probs[res1][res2][1][1].size(): " << dist_probs[res1][res2][1][1].size()<< std::endl;
                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }
                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));
						Real w_total = 0;
                        for(Size m = 1; m <= dist_probs[res1][res2][1][1].size() ; m++){
                            w_total += dist_probs[res1][res2][1][2][m];
                        }
						Real total_dist_diff2 = 0;
                        for(Size n = 1; n <= dist_probs[res1][res2][1][1].size() ; n++){
                            Real pdist = dist_probs[res1][res2][1][1][n];
                            Real y_weight = dist_probs[res1][res2][1][2][n] / w_total;
                            Real dist_diff = sqrt((dist - pdist) * (dist - pdist));
                            total_dist_diff2 += y_weight * dist_diff;
                        }
						// total_dist_diff_temp = epitope_prob[res2] * total_dist_diff2;
						// total_dist_diff2_final += total_dist_diff_temp;
						total_dist_diff2_final += total_dist_diff2;
                        count2++;
                    }
                }
            }
        }
    }

    if ( count2 == 0 )
		// total_dist_diff2 = 1000000.0;
        total_dist_diff2_final = 1000000.0;
    else
		// total_dist_diff2 /= count2;
        total_dist_diff2_final /= count2;

	// return total_dist_diff2;
    return total_dist_diff2_final;
}

Real DomainAssembly::score_rot_tra_v2( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff1 = 0;
    Real count1 = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr2_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");
						// 对蛋白质中两条链上的两个残基的α碳原子（CA）的坐标进行旋转和平移变换 由于rot_tra_1为6个0 所以rotMat1/traVec1都为0 没有对抗体链进行旋转平移
                        for ( Size k = 1; k <= 3; ++k ) {
							//抗体链上的残基在旋转平移变换后的坐标
                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
							//抗原链上的残基在旋转平移变换后的坐标
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }
						//计算旋转平移后它们之间的距离
                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr2_[j-1][res1][res2];
						//进而计算和一个已知的距离约束（pdist）之间的差异
                        total_dist_diff1 += sqrt((dist - pdist) * (dist - pdist));
                        count1++;
                    }
                }
            }
        }
    }

    if ( count1 == 0 )
        total_dist_diff1 = 1000000.0;
    else
        total_dist_diff1 /= count1;

    return total_dist_diff1;
}

Real DomainAssembly::score_rot_tra_v3( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff3 = 0;
    Real count3 = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr3_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr3_[j-1][res1][res2];

                        total_dist_diff3 += sqrt((dist - pdist) * (dist - pdist));
                        count3++;
                    }
                }
            }
        }
    }

    if ( count3 == 0 )
        total_dist_diff3 = 1000000.0;
    else
        total_dist_diff3 /= count3;

    return total_dist_diff3;
}

Real DomainAssembly::score_rot_tra_v4( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff4 = 0;
    Real count4 = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr4_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr4_[j-1][res1][res2];

                        total_dist_diff4 += sqrt((dist - pdist) * (dist - pdist));
                        count4++;
                    }
                }
            }
        }
    }

    if ( count4 == 0 )
        total_dist_diff4 = 1000000.0;
    else
        total_dist_diff4 /= count4;

    return total_dist_diff4;
}

Real DomainAssembly::score_rot_tra_v5( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff5 = 0;
    Real count5 = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr5_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr5_[j-1][res1][res2];

                        total_dist_diff5 += sqrt((dist - pdist) * (dist - pdist));
                        count5++;
                    }
                }
            }
        }
    }

    if ( count5 == 0 )
        total_dist_diff5 = 1000000.0;
    else
        total_dist_diff5 /= count5;

    return total_dist_diff5;
}

Real DomainAssembly::score_rot_tra_v6( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff6 = 0;
    Real count6 = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr6_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr6_[j-1][res1][res2];

                        total_dist_diff6 += sqrt((dist - pdist) * (dist - pdist));
                        count6++;
                    }
                }
            }
        }
    }

    if ( count6 == 0 )
        total_dist_diff6 = 1000000.0;
    else
        total_dist_diff6 /= count6;

    return total_dist_diff6;
}

utility::vector1< utility::vector1< Real > > DomainAssembly::matrix_multiply( utility::vector1< utility::vector1< Real > > arrA, utility::vector1< utility::vector1< Real > > arrB ) {

	//矩阵arrA的行数
	Size rowA = arrA.size();
	//矩阵arrA的列数
	Size colA = arrA[1].size();
	//矩阵arrB的行数
	Size rowB = arrB.size();
	//矩阵arrB的列数
	Size colB = arrB[1].size();
	//相乘后的结果矩阵
	utility::vector1< utility::vector1< Real > > res;
	if ( colA != rowB ) {//如果矩阵arrA的列数不等于矩阵arrB的行数。则返回空
		return res;
	}
	else {
		//设置结果矩阵的大小，初始化为为0
		res.resize( rowA );
		for ( Size i = 1; i <= rowA; ++i ) {
			res[i].resize( colB );
		}

		//矩阵相乘
		for ( Size i = 1; i <= rowA; ++i ) {
			for ( Size j = 1; j <= colB; ++j ) {
				for ( Size k = 1; k <= colA; ++k ) {
					res[i][j] += arrA[i][k] * arrB[k][j];
				}
			}
		}
	}
	return res;
}

//将三个欧拉角（X、Y、Z轴的旋转）转换为一个三维旋转矩阵（3×3 的 rotation matrix）
utility::vector1< utility::vector1< Real > > DomainAssembly::EulerAngles_to_rotationMatrix( Real euler_x, Real euler_y, Real euler_z ) {
	//由于三角函数要求单位是弧度 这里将角度转换为弧度
    euler_x = euler_x * 3.141592653589793 / 180.0;
    euler_y = euler_y * 3.141592653589793 / 180.0;
    euler_z = euler_z * 3.141592653589793 / 180.0;
	// 绕 X 轴旋转的矩阵 Rx
    utility::vector1< utility::vector1< Real > > R_x = {{1,            0,             0},\
                                                        {0, cos(euler_x), -sin(euler_x)},\
                                                        {0, sin(euler_x),  cos(euler_x)}};
	// 绕 Y 轴旋转的矩阵 Ry
    utility::vector1< utility::vector1< Real > > R_y = {{cos(euler_y),  0, sin(euler_y)},\
                                                        {0,             1,            0},\
                                                        {-sin(euler_y), 0, cos(euler_y)}};
	// 绕 Z 轴旋转的矩阵 Rz
    utility::vector1< utility::vector1< Real > > R_z = {{cos(euler_z), -sin(euler_z), 0},\
                                                        {sin(euler_z),  cos(euler_z), 0},\
                                                        {0,             0,            1}};
	// 矩阵相乘 先绕 X，再绕 Y
    utility::vector1< utility::vector1< Real > > R_y_x = matrix_multiply(R_y, R_x);
	// 然后绕 Z
    utility::vector1< utility::vector1< Real > > R_z_y_x = matrix_multiply(R_z, R_y_x);
	// 返回的是一个 3x3 的旋转矩阵
    return R_z_y_x;
}


utility::vector1< Real > DomainAssembly::spherical_to_translationVector( Real spher_r, Real spher_t, Real spher_p ) {

    spher_t = spher_t * 3.141592653589793 / 180.0;
    spher_p = spher_p * 3.141592653589793 / 180.0;

    Real x = spher_r * sin(spher_t) * cos(spher_p);
    Real y = spher_r * sin(spher_t) * sin(spher_p);
    Real z = spher_r * cos(spher_t);

    utility::vector1< Real > T;

    T.push_back(x);
    T.push_back(y);
    T.push_back(z);

    return T;
}


utility::vector1< utility::vector1< Real > > DomainAssembly::rot_tra_to_affine( utility::vector1< Real > rot_tra ) {

    Real euler_x(rot_tra[1]);
    Real euler_y(rot_tra[2]);
    Real euler_z(rot_tra[3]);
    Real spher_r(rot_tra[4]);
    Real spher_t(rot_tra[5]);
    Real spher_p(rot_tra[6]);

    utility::vector1< utility::vector1< Real > > rot_mat = EulerAngles_to_rotationMatrix( euler_x, euler_y, euler_z );
    utility::vector1< Real > tra_vec = spherical_to_translationVector( spher_r, spher_t, spher_p );

    utility::vector1< utility::vector1< Real > > affine_mat;

    affine_mat.resize( rot_mat.size() + 1 );
	for ( Size i = 1; i <= rot_mat.size() + 1; ++i ) {
		affine_mat[i].resize( rot_mat[1].size() + 1 );
	}

    for ( Size i = 1; i <= rot_mat.size(); ++i ) {
        for ( Size j = 1; j <= rot_mat[1].size(); ++j ) {
            affine_mat[i][j] = rot_mat[i][j];
        }
    }

    for ( Size i = 1; i <= tra_vec.size(); ++i ) {
        affine_mat[i][4] = tra_vec[i];
        affine_mat[4][i] = 0;
    }

    affine_mat[4][4] = 1;

    return affine_mat;
}


////////////////////////////////////////////////////////////////////////////////
/// @brief Transform the index of residue number (full to part)
Size DomainAssembly::index_f2p( Size f_index ) {

	Size p_index;
	Size num;

	for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
		if ( f_index >= movable_residues_[i].first && f_index <= movable_residues_[i].second ) {
			num = 0;
			for ( Size j = 1; j <= i-1; ++j ) {
				num += ( movable_residues_[j].second - movable_residues_[j].first + 1 );
			}
			num += ( f_index - movable_residues_[i].first + 1 );
			p_index = 2 * num - 1;

			return p_index;
		}
	}

	for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
		if ( f_index >= immovable_residues_[i].first && f_index <= immovable_residues_[i].second ) {
			num = 0;
			for ( Size j = 1; j <= i-1; ++j ) {
				num += ( immovable_residues_[j].second - immovable_residues_[j].first + 1 );
			}
			num += ( f_index - immovable_residues_[i].first + 1 );
			p_index = num;

			return p_index;
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Transform the index of residue number (part to full)
Size DomainAssembly::index_p2f( bool is_axis, Size p_index ) {

	Size f_index;
	Size num;

	if ( is_axis ) {
		num = 0;
		for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
			for ( Size j = movable_residues_[i].first; j <= movable_residues_[i].second; ++j ) {
				++num;
				if ( num == (p_index + 1) / 2 ) {
					f_index = j;
					return f_index;
				}
			}
		}
	} else {
		num = 0;
		for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
			for ( Size j = immovable_residues_[i].first; j <= immovable_residues_[i].second; ++j ) {
				++num;
				if ( num == p_index ) {
					f_index = j;
					return f_index;
				}
			}
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Accepted with Boltzmann probability
bool DomainAssembly::boltzmann_accept( const Real targetEnergy, const Real trialEnergy, Real recipocal_KT ) {

	if ( trialEnergy <= targetEnergy )
		return true;
	else {
		Real probability = exp( -(trialEnergy - targetEnergy) / recipocal_KT );
		if ( probability >= numeric::random::rg().uniform() )
			return true;
		else
			return false;
	}
}

bool DomainAssembly::greedy_accept( const Real targetEnergy, const Real trialEnergy ) {

	if ( trialEnergy <= targetEnergy )
		return true;
	else
		return false;
}


////////////////////////////////////////////////////////////////////////////////
/// @brief Given a pose, generate the constraints
core::scoring::constraints::ConstraintCOPs DomainAssembly::generate_constraints( Pose const & pose ) const {

	using namespace protocols::trRosetta;
	using namespace core::scoring::constraints;
	using namespace core::scoring::func;

	utility::vector1< core::scoring::constraints::ConstraintCOP > outputvec;

	Size const seqlength( pose.total_residue() );

	//Parameters
	Size const n_dist_bins_model( 37 ); //In neural net.
	Size const n_dist_bins( 35 ); //In constraints spline.
	Real const dist_bins_denominator( 19.5 );
	Real const dist_bins_exponent( 1.57 );
	utility::vector1< Real > dist_bins_vect( n_dist_bins );
	utility::vector1< Real > dist_bins_background( n_dist_bins - 3 );
	dist_bins_vect[1] = 0.0;
	dist_bins_vect[2] = 2.0;
	dist_bins_vect[3] = 3.5;

	for( Size ii(4); ii <= n_dist_bins; ++ii ) {
		dist_bins_vect[ii] = 4.25 + static_cast< Real >(ii-4)*0.5;
		dist_bins_background[ii-3] = std::pow( dist_bins_vect[ii]/dist_bins_denominator, dist_bins_exponent );
	}

	bool const pose_is_centroid( pose.residue_type_set_for_pose()->mode() == core::chemical::CENTROID_t );

	//Looping over every pair of residues
	for( Size ires(1); ires < seqlength; ++ires ) {
		core::chemical::ResidueType const & restype_i( pose.residue_type(ires) );
		if( !restype_i.is_canonical_aa() ) continue;
		core::id::AtomID const ca_atom_i( restype_i.atom_index( "CA" ), ires );

		for( Size jres(ires+1); jres <= seqlength; ++jres ) {
			core::chemical::ResidueType const & restype_j( pose.residue_type(jres) );
			if( !restype_j.is_canonical_aa() ) continue;
			core::id::AtomID const ca_atom_j( restype_j.atom_index( "CA" ), jres );

			bool intra_domain_flag = false;
			for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
				if ( ires >= immovable_residues_[i].first && ires <= immovable_residues_[i].second && jres >= immovable_residues_[i].first && jres <= immovable_residues_[i].second )
					intra_domain_flag = true;
			}

			if ( !intra_domain_flag ) {

				if( set_generate_dist_constraints_ ) {
					 Real prob_cutoff_;
					if ( !relax_ )
						prob_cutoff_ = dist_prob_cutoff_;
					else
						prob_cutoff_ = ref_dist_prob_cutoff_;
					generate_dist_constraints(
						outputvec, dist_bins_background, dist_bins_vect,
						ca_atom_i, ca_atom_j,
						n_dist_bins, n_dist_bins_model, ires, jres, prob_cutoff_
					);
				}
			}
		}
	} //End loop over every pair of residues.

	return outputvec;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Generate the distance constraints
void DomainAssembly::generate_dist_constraints(
	utility::vector1< core::scoring::constraints::ConstraintCOP > & outputvec,
	utility::vector1< Real > const & dist_bins_background,
	utility::vector1< Real > const & dist_bins_vect,
	core::id::AtomID const & ca_atom_i,
	core::id::AtomID const & ca_atom_j,
	Size const n_dist_bins,
	Size const n_dist_bins_model,
	Size const ires,
	Size const jres,
	Real const prob_cutoff
) const {

	using namespace protocols::trRosetta;
	using namespace core::scoring::constraints;
	using namespace core::scoring::func;

	Real dist_probsum(0.0);
	for( Size ibin(6); ibin <= n_dist_bins_model; ++ibin ) {
		dist_probsum += dist_constraints[ires][jres][ibin];
	}

	if( dist_probsum <= prob_cutoff ) return; //Can stop here if below probability cutoff.

// 	if( TR.Debug.visible() ) {
// 		TR.Debug << "Generating distance constraints between residues " << ires << " and " << jres << "." << std::endl;
// 	}

	utility::vector1< Real > dist_attractive_repulsive(n_dist_bins);
	for( Size ibin(6); ibin <= n_dist_bins_model; ++ibin ) {
		dist_attractive_repulsive[ibin-2] = -std::log( (dist_constraints[ires][jres][ibin] + 0.0001) / (dist_constraints[ires][jres][n_dist_bins_model] * dist_bins_background[ibin-5] ) ) - 0.5;
	}
	Real const addval( std::max( 0.0, dist_attractive_repulsive[4] ) );
	dist_attractive_repulsive[1] = addval + 10.0;
	dist_attractive_repulsive[2] = addval + 3.0;
	dist_attractive_repulsive[3] = addval + 0.5;

// 	if( TR.Debug.visible() ) {
// 		TR.Debug << "x_axis\t";
// 		for( Size ii(1); ii <= dist_bins_vect.size(); ++ii) {
// 			TR.Debug << dist_bins_vect[ii] << "\t";
// 		}
// 		TR.Debug << std::endl;
// 		TR.Debug << "y_axis\t";
// 		for( Size ii(1); ii <= dist_attractive_repulsive.size(); ++ii) {
// 			TR.Debug << dist_attractive_repulsive[ii] << "\t";
// 		}
// 		TR.Debug << std::endl;
// 	}

	SplineFuncOP splinefunc(
		utility::pointer::make_shared< SplineFunc >(
			"TAG", distance_constraint_weight_, 1.0, 0.5,
			dist_bins_vect, dist_attractive_repulsive
		)
	);
	outputvec.push_back( utility::pointer::make_shared< AtomPairConstraint >( ca_atom_i, ca_atom_j, splinefunc ) );
}

/// @brief Remove previously-added constraints from the pose.
void DomainAssembly::remove_constraints(
	utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
	Pose & pose
) const {
	runtime_assert_string_msg( pose.remove_constraints( constraints, true ), "Program error in DomainAssembly::remove_constraints(): Something went wrong when trying to remove constraints." );
}

/// @brief Add constraints from a list to the pose, if the constraints are between residues that are
/// separated by at least min_seqsep but less than max_seqsep residues.
/// @details This does not clear constraints from the pose.
/// @note It is "less than" max_seqsep, not "less than or equal to".
void DomainAssembly::add_constraints_to_pose(
	Pose & pose,
	utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
	Size const min_seqsep,
	Size const max_seqsep,
	bool const skip_glycine_positions /*=false*/
) const {
	for ( auto const & cst : constraints ) {
		Size res1, res2;
		get_residues_from_constraint( res1, res2, cst );
//		if ( skip_glycine_positions ) {
//			if (
//					pose.residue_type(res1).aa() == core::chemical::aa_gly ||
//					pose.residue_type(res2).aa() == core::chemical::aa_gly
//					) {
//				continue;
//			}
//		}
		debug_assert(res2 > res1);
		Size const seqdist( res2 - res1 ); //Assumes res2>res1
		if ( seqdist >= min_seqsep && seqdist < max_seqsep ) {
			pose.add_constraint( cst );
		}
	}
}

/// @brief Given a constraint, determine if it is an AtomPairConstraint, an
/// AngleConstraint, or a DihedralConstraint, pull out the pair of residues
/// that are constrained, and return the pair.
/// @details Throws if type is unrecognized or if more than two residues are
/// constrained.  Values of res1 and res2 are overwritten by this operation.
void DomainAssembly::get_residues_from_constraint(
	Size & res1,
	Size & res2,
	core::scoring::constraints::ConstraintCOP const & cst
) const {
	if ( get_residues_from_atom_pair_constraint( res1, res2, cst ) ) {
		return;
	}
//	if ( get_residues_from_angle_constraint( res1, res2, cst ) ) {
//		return;
//	}
//	if ( get_residues_from_dihedral_constraint( res1, res2, cst ) ) {
//		return;
//	}
	utility_exit_with_message( "DomainAssembly::get_residues_from_constraint(): Selected constraint was not an AtomPairConstraint!" );
}

/// @brief Given a constraint, determine if it is an AtomPairConstraint pull
/// out the pair of residues that are constrained, and return the pair.
/// @details If successful, values of res1 and res2 are overwritten by this
/// operation, and the function returns "true".  Otherwise, values are not
/// altered, and the function returns "false".
bool DomainAssembly::get_residues_from_atom_pair_constraint(
	Size & res1,
	Size & res2,
	core::scoring::constraints::ConstraintCOP const & cst
) const {
	using namespace core::scoring::constraints;
	AtomPairConstraintCOP atpair_cst( utility::pointer::dynamic_pointer_cast< AtomPairConstraint const >(cst) );
	if ( atpair_cst == nullptr ) return false;
	res1 = atpair_cst->atom1().rsd();
	res2 = atpair_cst->atom2().rsd();
	if ( res1 > res2 ) {
		std::swap(res1, res2);
	}
	return true;
}

/// @brief Do all-atom FastRelax refinement.
void DomainAssembly::do_fullatom_refinement( core::pose::Pose & pose, utility::vector1< bool > move_bb ) const {

	core::kinematics::MoveMapOP movemap( utility::pointer::make_shared< core::kinematics::MoveMap >() );
	movemap->set_bb(move_bb);
	movemap->set_chi(false);
	movemap->set_jump(true);

	protocols::relax::FastRelaxOP frlx( utility::pointer::make_shared< protocols::relax::FastRelax >() );
	frlx->set_scorefxn(sfxn_fullatom_);
	frlx->dualspace(true);
	frlx->set_movemap(movemap);

	frlx->apply(pose);
}

void DomainAssembly::Pareto_method1(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{
		E[0] = population_score_[i];
	
		E[1] = score_rot_tra_v2(target_rot_tra[i]);
		pareto.addvalue(E, i);
	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
	}
}

void DomainAssembly::Pareto_method2(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{
	
		E[0] = population_score_[i];
	
//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v1(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

void DomainAssembly::Pareto_method3(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{
		
		E[0] = population_score_[i];
		
//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v1(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

void DomainAssembly::Pareto_method4(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{

		E[0] = population_score_[i];
	
//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v1(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

void DomainAssembly::Pareto_method5(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{
	
		E[0] = population_score_[i];

//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v1(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

void DomainAssembly::Pareto_method6(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score_.size(); i++)
	{
		
		E[0] = population_score_[i];
		
//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v1(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

utility::vector1<utility::vector1<utility::vector1<Real>>> DomainAssembly::delete_unwanted_rot_tra(utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Size> &target_index)
{
//	TR << "delete not PARETO pose" << std::endl;
    utility::vector1< utility::vector1< utility::vector1< Real > > > temp_rot_tra;

	for (Size i = 1; i <= target_index.size(); i++)
	{
		temp_rot_tra.emplace_back(target_rot_tra[target_index[i]]);
	}

	utility::vector1< utility::vector1< utility::vector1< Real > > >().swap(target_rot_tra);

	for (Size i = 1; i <= target_index.size(); i++)
	{
		target_rot_tra.emplace_back(temp_rot_tra[i]);
	}
	utility::vector1< utility::vector1< utility::vector1< Real > > >().swap(temp_rot_tra);

    return target_rot_tra;

}

using DomainAssemblyOP = utility::pointer::shared_ptr<DomainAssembly>;

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// MAIN /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int
main( int argc, char * argv [] )
{
	try {
		// initialize option system, RNG, and all factory-registrators
		devel::init(argc, argv);

		//  protocols::jd2::register_options();

		// Create and kick off a new load mover
		core::pose::Pose pose;
		DomainAssemblyOP domain_assembly( new DomainAssembly() );
		protocols::jd2::JobDistributor::get_instance()->go( domain_assembly );

		return 0;

	}
	catch (utility::excn::Exception const & e ) {
		e.display();
		return -1;
	}
}
